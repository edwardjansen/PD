# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:53:48 2019

@author: EdwardJansenADC
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:00:29 2019

@author: Edward
"""

import os
print(os.getcwd())

import numpy as np
import pandas as pd


def variable_prep(df, ordinals, cls_lbl, standardize = False):
    """
    Function converses ordinal variables to discrete numbers, dummy encodes nominal variables,
    and standardizes numerical values if standardize=True. 
    ---------
    Input:
        df:         Pandas dataframe
        ordinals:   array; containing the indices of the ordinal variables in df
        cls_lbl:    scalar; contains index of dependent variable in df
    
    Returns:
        var_type    prints the variable type
        data        new Pandas dataframe containing encoded vars
    
    """
    from sklearn.preprocessing import LabelEncoder 
    from sklearn.preprocessing import StandardScaler 
    data = df.copy(deep=True)    
    var_type = [0]*len(data.columns)
    
    # determine variable type
    for i in range(0, len(data.columns)):
        if data[data.columns[i]].dtype.kind in 'bifc':    
            var_type[i] = 'numerical'
        elif i in ordinals:
            var_type[i] = 'ordinal'
        elif i == cls_lbl:
            var_type[i] = 'ordinal'    #strictly speaking, this is not an ordinal value, but the encoding works simpler this way
        else:
            var_type[i] = 'nominal'
    print(var_type)
    
    # transform data depending on variable type
    old_vars = data.columns
    for i in range(0,len(var_type)):
        if var_type[i] == 'ordinal':
            class_le = LabelEncoder()
            data[old_vars[i]] = class_le.fit_transform(data[old_vars[i]].values)
        elif var_type[i] == 'nominal':
            new_cols = pd.get_dummies(data[old_vars[i]]).columns
            data[new_cols] = pd.get_dummies(data[old_vars[i]])
            data = data.drop(old_vars[i], axis = 1)
        elif var_type[i] == 'numerical' and standardize == True:
            stdsc = StandardScaler() 
            data[old_vars[i]] = stdsc.fit_transform(data[old_vars[i]]) 
    
    return data



def binning(df, target, exception, bins=10, labels=None):
    """
    Function binnes a Pandas dataframe using .qcut(), which tries to make bins of equal frequency.
    ---------
    Input:
        df:         dataframe;
        target:     string; name of dependent variable
        bins:       list; number of ideal bins
    
    Returns:
        binned_df   dataframe; the binned dataframe
        retbins     list; list of bins per variable
        lbl         list; list of variables the algorithm does not bin
    
    """
    data = df.copy(deep=True)
    cols_ = data.columns[~data.columns.isin([target])] 
    cols = cols_[~cols_.isin(exception)]
    binned_df = pd.DataFrame(columns = cols)
    retbins = []
    lbl = []
    
    # bin if number of unique vals / per > 10, otherwise do nothing
    for i in cols[~cols.isin([target])]:
        if (len(np.unique(data[i]))>15):
            binned_df[i], tmp = pd.qcut(data[i], bins, retbins=True, duplicates='drop', labels=labels)
            retbins.append(tmp)
        else:
            binned_df[i] = data[i]
            tmp = len(binned_df[i].unique())
            retbins.append(tmp)
            lbl.append(i)
            
        binned_df[target] = data[target]
        
    return binned_df, retbins, lbl



def binning_setbins(df, target, bins, labels=[], include_lowest=True, correct_nan = False):
    """
    Function binnes a Pandas dataframe using .cut(). Can be used to bin the test set to the same bins as the 
    train set.
    ---------
    Input:
        df:         dataframe;
        target:     string; name of dependent variable
        bins:       list; list of bins
    
    Returns:
        binned_df   dataframe; the binned dataframe
    
    """
    data = df.copy(deep=True)
    cols = data.columns[~data.columns.isin([target])] 
    binned_df = pd.DataFrame(columns = cols) 
    
    # bin using .cut() for every column in df
    for i in range(len(cols)):
        if cols[i] in labels:
            binned_df[cols[i]] = data[cols[i]]
        else:
            binned_df[cols[i]] = pd.cut(data[cols[i]], bins=bins[i], labels=None, include_lowest=include_lowest)
    
    # put outliers in the outer left or right most bin
    for i in cols:
        if correct_nan == True:
            idx = binned_df[i].index[ binned_df[i].isnull() == True].tolist()
            for j in range(len(idx)):
                if data[i][idx[j]] > np.max( bins[ cols.tolist().index(i) ] ):
                    binned_df[i][idx[j]] = binned_df[i].dropna().max()
                elif data[i][idx[j]] < np.min( bins[ cols.tolist().index(i) ] ):
                    binned_df[i][idx[j]] = binned_df[i].dropna().min()
            #binned_df[i] = binned_df[i].fillna(binned_df[i].mode()[0])
        
    binned_df[target] = data[target]
    
    return binned_df



def manual_binning(df, target, bins, labels=None, include_lowest=True):
    """
    Function binnes a single variable .cut() and shows the WoE for the new bins.
    Can be used to manually change binning boundaries and see result for the WoE.    
    ---------
    Input:
        df:         dataframe; 2-column Pandas dataframe containing the to be binned variable 
                               and dependent variable
        target:     string; name of dependent variable
        bins:       list; list of bins
    
    Returns:
        cut_df      dataframe; the newly binned single variable
    
    """
    data_target = df[target].copy(deep=True)
    data = df[ df.drop([target], axis=1).columns[0] ]
    
    # bin using .cut() according to bin-array specified
    cut_df = pd.cut(data, bins=bins, labels=labels, include_lowest=include_lowest)

    # compute WoE of each category of newly binned var
    df0 = pd.DataFrame({'x': cut_df, 'y': data_target})
    d = df0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
    d.columns = ['Cutoff', 'N', 'Events']
    d['% of Events'] = (d['Events']+0.5) / d['Events'].sum()  #prevent /0
    d['Non-Events'] = d['N'] - d['Events']
    d['% of Non-Events'] = (d['Non-Events']+0.5)/ d['Non-Events'].sum() #prevent /0
    d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])

    # compute IV for each category
    d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events']) 

    # plot WoE trend per category
    ax = d.plot.bar(x='Cutoff', y='WoE')
    
    return cut_df



def test_woe(df, target, trend_visualize=True):
    """
    Function computes WoE en IV values for all variables in the input dataframe that should already be binned.  
    ---------
    Input:
        df:         dataframe; dataframe containing the binned variables
        target:     string; name of dependent variable
    
    Returns:
        IV plot     plot; plots IV values per variable
        WoE plot    plot; plots WoE per bin for every variable
        cols        pd index; array with column names
        d_lst       list; list containing bin statistics including WoE & IV
        df_IV       dataframe; df containing IV per variable
    
    """
    data = df.copy(deep=True)
    cols = data.columns[~data.columns.isin([target])]   
    d_lst = []
    df_IV = pd.DataFrame()
   
    # Compute WoE and IV per category for each variable 
    for i in cols:
        df0 = pd.DataFrame({'x': data[i], 'y': data[target]})
        
        d = df0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = (d['Events']+0.5) / d['Events'].sum()  #prevent /0
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = (d['Non-Events']+0.5)/ d['Non-Events'].sum() #prevent /0
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])

        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events']) 
        d_lst.append(d)
        temp = pd.DataFrame({"Variable" : [i], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        df_IV = pd.concat([df_IV,temp], axis=0) 
    
    # create table with sorted IV per variable
    df_IV = df_IV.sort_values('IV', axis=0, ascending=False)   
    df_IV.plot.bar(x='Variable', y = 'IV')
    
    # show plots of WoE per category for each variable    
    if trend_visualize == True:
        for i in range(len(cols)):
            ax = d_lst[i].plot.bar(x='Cutoff', y='WoE')
            ax.set_title(cols[i]) 
   
    return cols, d_lst, df_IV



def replace_woe(df, d_lst, target):
    """
    Function replaces bin values with WoE values.  
    ---------
    Input:
        df:         dataframe; dataframe containing the binned variables
        d_lst:      list; list with the bin statistics of which the WoE value is used
        target:     string; name of dependent variable
    
    Returns:
        woe_df      dataframe; dataframe containing the df with WoE values for all variables
    
    """
    data = df.copy(deep=True)    
    cols = data.columns[~data.columns.isin([target])] 
    woe_df = pd.DataFrame()

    # for each variable i, replace the bin values in data with the WoE values in d_lst
    k=0
    for i in cols:
        for j in range(len(d_lst[k])):
            data.loc[  data[i].eq(d_lst[k]['Cutoff'][j]), 'woe_'+str(i)  ] = d_lst[k]['WoE'][j]
        k+=1
    woe_cols = ['woe_'+str(cols[i]) for i in range(len(cols))]
    woe_df = data[woe_cols]
    woe_df = pd.concat([woe_df, data[target]], axis=1)
    
    return woe_df



def scorecard(d_lst, cols, coef, intercept, target_score=600, target_odds = 50, pts_double_odds=20):
    """
    Function that creates a scorecard based on the binning and corresponding WoE values and
    the result of the model fitting.  
    ---------
    Input:
        d_lst:      list; list with the bin statistics of which the WoE value is used
        cols:       pd index; list with variable names (omitting dependent variable)
        coef:       np array; list containing fitted regression coefficients
        intercept:  scalar; intercept of regression
    
    Returns:
        scorecard   dataframe; scorecard dataframe
    
    """
    # set up scorecard and basic scaling parameters
    scorecard = pd.DataFrame()
    factor = pts_double_odds / np.log(2)
    offset = target_score - factor * np.log(target_odds)
    
    # create list of Attributes with their corresponding WoE
    for i in range(len(cols)):
        df = pd.DataFrame({'Characteristic': cols[i] , 'Attribute': d_lst[i]['Cutoff'], 'WoE': d_lst[i]['WoE']})
        scorecard = pd.concat([scorecard, df], axis=0)
    
    # add to above list the score per attribute based on WoE and coefficients for logreg
    scorecard['Score'] = np.zeros(len(scorecard))
    for i in range(len(cols)):
        for j in range(0, len(scorecard)):
            if cols[i] == scorecard['Characteristic'].iloc[j]:
                scorecard['Score'].iloc[j] = np.round((coef[i]*scorecard['WoE'].iloc[j] + intercept/len(coef))*factor+ offset/len(coef),0)
    
    return scorecard



def correlation_check(df, threshold, bins=10, figsave=False):
    """
    Function that plots the distribution of correlation coefficients between the independent variables and 
    finds the coefficients (and their indices) that are bigger than a certain threshold.  
    ---------
    Input:
        df          dataframe; dataframe with (WoE values of) independent variables 
        threshold   scalar; maximal allowed correlation between 'independent' variables
    
    Returns:
        corr plot   plot; plot of distribution of correlation coefficients
        [c., x, y]  tuple; tuple containing the larger than threshold coefficients with corresponding x,y
                           indices in the correlation matrix.
    
    """
    # compute correlation matrix, extract upper triangle, find coefficients and corresponding variables
    corr = np.triu(df.corr(),1)
    idx,idy = np.where(corr>threshold)
    corrcoef = corr[idx, idy]
    x = df.columns[idx]
    y = df.columns[idy]

    # plot distribution of corr coefficients
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(corr.flatten(), bins=bins, normed=False)
    ax.set_xlim([-1,1])
    ax.set_ylim([0,30])

    if figsave == True:
        plt.savefig(r'corrcoef_hist.pdf')
        
    return [corrcoef, x, y]



def GS_clf(penalty = ['l1', 'l2'], C = np.logspace(-7, 7, 29), cv=5):
    """
    Function that sets up the GridSearch model with crossvalidation.
    ---------
    Input:

    Returns:
        clf         object; to be fitted classifier
    
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    
    # set up logistic regression
    lr = LogisticRegression(C=1)
    
    # define grid of hyper parameters
    param_grid = [
            {'penalty' : ['l1', 'l2'],
            'C' : np.logspace(-7, 7, 29),
            'solver' : ['liblinear']},
        ]    
    
    # initialize classifier
    clf = GridSearchCV(lr, param_grid, scoring = 'roc_auc', n_jobs=-1, iid =True, refit = True, 
                           cv=cv, verbose=True, error_score='raise')
    
    return clf

def split(df, target, test_size = 0.25):
    """
    Function that splits a dataset into train and test set.
    ---------
    Input:
        df          dataframe; to be splitted data   
        target:     string; name of dependent variable        
    Returns:
        X_train     dataframe; train data
        X_test      dataframe; test data
        y_train     array; train labels
        y_test      array; test labels
    
    """
    from sklearn.model_selection import train_test_split
    
    # split in train and test
    X_train, X_test, y_train, y_test = train_test_split(df.drop([target], axis=1), 
                                                        df[target],
                                                        test_size = test_size,
                                                        random_state=101)
    
    return X_train, X_test, y_train, y_test



def gini(y_true, y_pred):
    """
    Function computes the Gini coefficient between true labels and predicted probabilities.
    ---------
    Input:
        y_true      array; true labels
        y_pred      array; predicted probabilities       
    Returns:
        G_norm      scalar; Gini coefficient
    
    """
    # check for incorrect input
    assert y_true.shape == y_pred.shape
    n_samples = len(y_true)
    arr = np.array([y_true, y_pred]).transpose()
    
    # sort rows on prediction column (from smallest to largest)
    true_order = arr[arr[:,0].argsort()][:,0]
    pred_order = arr[arr[:,1].argsort()][:,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)    

    # get Gini coefficients (area between curves)
    G_true = 1-np.sum(np.diff(L_ones,1)*(np.roll(L_true,-1)+L_true)[0:-1])
    G_pred = 1-np.sum(np.diff(L_ones,1)*(np.roll(L_pred,-1)+L_pred)[0:-1])
    G_norm = G_pred / G_true # the closer to 1, the more accurate the model!
    
    return G_norm
 
    
def var_type(df):
    """
    Function that returns whether the variables in a dataframe or numbers or symbols.
    ---------
    Input:
        df          dataframe; dataset 
    Returns:
        var_type    array; array containing the variable types
        idx_cat     array; indices of symbolic vars
    
    """
    # check whether variable is numerical or symbolic
    var_type = ['number' if df[i].dtype.kind in 'bifc' else 'symbol' for i in df.columns ]
    
    # find indices of symbolic vars for further processing
    idx_cat = []    
    for i in range(len(var_type)):
        if var_type[i] == 'symbol':
            tmp = i
            idx_cat.append(tmp)
        
    return var_type, idx_cat



def score_data(woe_df, coef, intercept, target_score=600, target_odds = 50, pts_double_odds=20, visualize=True, savefig=False):
    """
    Function that scores a dataset based on the model. It is optional to output a distribution plot. 
    ---------
    Input:
        woe_df:     dataframe; WoE transformed df
        coef:       np array; list containing fitted regression coefficients
        intercept:  scalar; intercept of regression
    
    Returns:
        woe_df      dataframe; scorecard dataframe
    
    """
    import matplotlib.pyplot as plt
    
    # set up scoring parameters
    factor = pts_double_odds / np.log(2)
    offset = target_score - factor * np.log(target_odds)
    
    # compute score of each datapoint
    woe_df['Score'] = np.zeros(len(woe_df))
    for i in range (0,len(coef)):
        woe_df['Score'] += ((woe_df[woe_df.columns[i]] * coef[i])+ intercept/len(coef))*factor+ offset/len(coef)

    # compute log odds, odds and probs    
    woe_df['log odds'] = (woe_df['Score'] - offset) / factor
    woe_df['odds'] = np.exp(woe_df['log odds'])
    woe_df['probs'] = woe_df['odds'] / (woe_df['odds'] + 1)
    
    if visualize == True:
        # Add Scoring Groups to Plot
        plt.figure(figsize=(10, 5))
        count, score, _ = plt.hist(woe_df['Score'],
                 bins='auto',
                 edgecolor='white',
                 color = '#317DC2',
                 linewidth=1.2)
        
        plt.xlim(np.min(score),np.max(score))
        plt.title('Scoring Groups', fontweight="bold", fontsize=14)
        plt.xlabel('Score')
        plt.ylabel('Count')
        
        # Percentile Lines
        plt.axvline(np.percentile(woe_df['Score'],25), color='red', linestyle='dashed', linewidth=2, alpha=0.6)
        plt.axvline(np.percentile(woe_df['Score'],50), color='orange', linestyle='dashed', linewidth=2, alpha=0.6)
        plt.axvline(np.percentile(woe_df['Score'],75), color='green', linestyle='dashed', linewidth=2, alpha=0.6)
        
        # Text
        plt.text(np.min(score)+5, np.max(count), 'Poor', color='red', fontweight='bold', style='italic', fontsize=12)
        plt.text(np.percentile(woe_df['Score'],25)+5, np.max(count), 'Fair', color='orange', fontweight='bold', style='italic', fontsize=12)
        plt.text(np.percentile(woe_df['Score'],50)+5, np.max(count), 'Good', color='green', fontweight='bold', style='italic', fontsize=12)
        plt.text(np.percentile(woe_df['Score'],75)+5, np.max(count), 'Excellent', color='blue', fontweight='bold', style='italic', fontsize=12)
        
        # Shading between Percentiles
        plt.axvspan(np.min(score), np.percentile(woe_df['Score'],25), alpha=0.1, color='red')
        plt.axvspan(np.percentile(woe_df['Score'],25), np.percentile(woe_df['Score'],50), alpha=0.1, color='yellow')
        plt.axvspan(np.percentile(woe_df['Score'],50), np.percentile(woe_df['Score'],75), alpha=0.1, color='green')
        plt.axvspan(np.percentile(woe_df['Score'],75), np.max(score), alpha=0.1, color='blue');
        
        if savefig == True:
            plt.savefig('Score_distribution.pdf')

    return woe_df
