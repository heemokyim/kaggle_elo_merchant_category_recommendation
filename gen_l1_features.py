# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:11:57 2018

@author: bcheung
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder 
from scipy.stats import entropy, kurtosis, skew

DATE_PERCENTILES = np.append(np.percentile(np.arange(0,31),np.arange(0,100,20)),np.inf)

def rolling_window(a,window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def calc_rolling_entropy(series,window_size,column):
    rolling_entropy = np.apply_along_axis(entropy,1,rolling_window(series, window_size)[-window_size:])
    rolling_entropy_cols = ['{}_w{}_r{}_entropy'.format(column,x,window_size) for x in np.arange(0,window_size)]
    rolling_entropy_dict = dict(zip(rolling_entropy_cols,rolling_entropy))
    return(rolling_entropy_dict)

def calc_rolling_sequence(series,window_size,column):
    rolling_values = rolling_window(series, window_size)[-window_size:]
    
    def combine_integers(series):
        return(''.join(series.astype('str')))
    
    rolling_seq = np.apply_along_axis(combine_integers,1,rolling_values)
    rolling_seq_cols = ['{}_w{}_r{}_seq'.format(column,x,window_size) for x in np.arange(0,window_size)]
    rolling_seq_dict = dict(zip(rolling_seq_cols,rolling_seq))
    return(rolling_seq_dict)
    
def calc_rolling_function(series,window_size,column,function):
    function_name = function.__name__
    rolling_function = np.apply_along_axis(function,1,rolling_window(series, window_size)[-window_size:])
    rolling_function_cols = ['{}_w{}_r{}_{}'.format(column,x,window_size,function_name) for x in np.arange(0,window_size)]
    rolling_function_dict = dict(zip(rolling_function_cols,rolling_function))
    return(rolling_function_dict)

def calc_cat_stats(df_series,window_size):
#    df_series = df_grp[col].copy()
#    window_size = 10
    
    #Extract the column name
    col = df_series.name
    
    encoded_values = LabelEncoder().fit_transform(df_series)
    
    cat_feats = {}
    cat_feats['nunique_{}'.format(col)] = df_series.nunique()
    cat_feats['entropy_{}'.format(col)] = entropy(LabelEncoder().fit_transform(df_series))
    
    rolling_entropy_feats = calc_rolling_entropy(encoded_values,window_size,col)
    rolling_sequence_feats = calc_rolling_sequence(encoded_values,window_size,col)
    
    full_cat_feats = {**cat_feats,
                      **rolling_entropy_feats,
                      **rolling_sequence_feats}
    
    return(full_cat_feats)
    
def calc_num_stats(df_series,window_size,q_values=np.arange(5,100,20)):
    
    col = df_series.name
    
    num_feats = {}
    num_feats['mean_{}'.format(col)] = np.mean(df_series)
    num_feats['std_{}'.format(col)] = np.std(df_series)
    num_feats['min_{}'.format(col)] = np.min(df_series)
    num_feats['max_{}'.format(col)] = np.max(df_series)
    num_feats['skew_{}'.format(col)] = skew(df_series)
    num_feats['kurt_{}'.format(col)] = kurtosis(df_series)
    
    percentile_values = np.percentile(df_series,q_values)
    percentile_cols = ['q{}_{}'.format(q,col) for q in q_values]
    percentile_feats = dict(zip(percentile_cols,percentile_values))
    
    rolling_mean_feats = calc_rolling_function(df_series,window_size,col,np.mean)
    rolling_std_feats = calc_rolling_function(df_series,window_size,col,np.std)
    rolling_min_feats = calc_rolling_function(df_series,window_size,col,np.min)
    rolling_max_feats = calc_rolling_function(df_series,window_size,col,np.max)
    rolling_kurtosis_feats = calc_rolling_function(df_series,window_size,col,kurtosis)
    rolling_skew_feats = calc_rolling_function(df_series,window_size,col,skew)
    
    full_num_feats = {**num_feats,
                      **percentile_feats,
                      **rolling_mean_feats,
                      **rolling_std_feats,
                      **rolling_min_feats,
                      **rolling_max_feats,
                      **rolling_kurtosis_feats,
                      **rolling_skew_feats}
    return(full_num_feats)
    
def calc_date_stats(df_series,window_size):
    
#    df_series = df_grp[col].copy()
#    window_size = 10
    
    col = df_series.name
    
    month_series = df_series.dt.month
    month_series.name = '{}_month'.format(col)
    day_series = df_series.dt.day
    day_series = pd.cut(day_series,DATE_PERCENTILES)
    day_series.name = '{}_day'.format(col)
    
    month_feats = calc_cat_stats(month_series,window_size)
    day_feats = calc_cat_stats(day_series,window_size)
    
    full_date_feats = {**month_feats,
                      **day_feats}
    
    return(full_date_feats)

def calc_overall_feats(df_grp,sort_key,feature_types_dict,feature_windows_dict):
    
#    df_grp = df_grp.copy()
#    sort_col = 'purchase_date'
#    feature_types_dict = HT_FEATURE_TYPES
#    feature_windows_dict = HT_FEATURE_WINDOWS
    
    df_grp = df_grp.sort_values([sort_key]).reset_index(drop=True)
    
    feat_list = {}
    
    for col in df_grp.columns:
        
        df_series = df_grp[col]
        window_size = feature_windows_dict[col]
        window_size = min(len(df_series),window_size)
        
        if feature_types_dict[col] == 'c':
            feats = calc_cat_stats(df_series,window_size)
        elif feature_types_dict[col] == 'n':
            feats = calc_num_stats(df_series,window_size)
        elif feature_types_dict[col] == 'd':
            feats = calc_date_stats(df_series,window_size)
        elif feature_types_dict[col] == 'cn':
            cat_feats = calc_cat_stats(df_series,window_size)
            num_feats = calc_num_stats(df_series,window_size)
            feats = {**cat_feats,**num_feats}
        elif feature_types_dict[col] == 'k':
            next
        else:
            raise RuntimeError('A column has not been mapped to data type')
        feat_list.update(feats)
    return(pd.Series(feat_list))
    
def gen_grouped_features(df,groupby_key,sort_key,feature_types_dict,feature_windows_dict,header):
#    df = new_merch_trans.copy()
#    groupby_key='card_id'
#    sort_key='purchase_date'
#    feature_types_dict=NM_FEATURE_TYPES
#    feature_windows_dict=NM_FEATURE_WINDOWS
#    header='nm'
#    
#    for key, df_grp in df.groupby(groupby_key):
#        break
#        features = calc_overall_feats(df_grp,sort_key,feature_types_dict,feature_windows_dict)
    
    features_grp = df.groupby(groupby_key).apply(lambda x: calc_overall_feats(x,sort_key,feature_types_dict,feature_windows_dict))
    if type(features_grp) == pd.core.series.Series:
        features_grp = pd.DataFrame(features_grp)
        features_grp = features_grp.reset_index()
        features_grp.columns=[groupby_key,'feature','value']
        features_grp = features_grp.set_index([groupby_key,'feature']).unstack('feature')
        features_grp.columns = features_grp.columns.droplevel(0)
    
    seq_cols = [cols for cols in features_grp.columns if len(re.findall('_seq$',cols)) >0]
    features_grp[seq_cols] = features_grp[seq_cols].astype(str)
    features_grp.columns = ['{}_{}'.format(header,c) for c in features_grp.columns]
    return(features_grp)

if __name__ == '__main__':
    
    features_grp = gen_grouped_features(hist_trans,'card_id','purchase_date',HT_FEATURE_TYPES,HT_FEATURE_WINDOWS)
    features_grp = gen_grouped_features(new_merch_trans,'card_id','purchase_date',NM_FEATURE_TYPES,NM_FEATURE_WINDOWS,'nm')
    
    
    
    