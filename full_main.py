# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:59:23 2018

@author: bcheung
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import itertools
import re
import datetime

from functools import partial

from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import entropy
from utils import parallelize_dataframe
from gen_l1_features import gen_grouped_features

HT_IMPUTE_DICT = {'category_3':'MISSING',
                  'merchant_id':'MISSING',
                  'category_2':'MISSING',
                  'most_recent_sales_range':'MISSING', 
                  'most_recent_purchases_range':'MISSING',
                  'avg_sales_lag3':-1, 
                  'avg_purchases_lag3':-1, 
                  'active_months_lag3':-1,
                  'avg_sales_lag6':-1, 
                  'avg_purchases_lag6':-1, 
                  'active_months_lag6':-1,
                  'avg_sales_lag12':-1, 
                  'avg_purchases_lag12':-1, 
                  'active_months_lag12':-1,
                  'category_4':'N'}

HT_FEATURE_TYPES = {'authorized_flag':'c',
                 'card_id':'k',
                 'city_id':'c',
                 'category_1':'c',
                 'installments':'n',
                 'category_3':'c',
                 'merchant_category_id':'c',
                 'merchant_id':'c',
                 'month_lag':'n',
                 'purchase_amount':'n',
                 'purchase_date':'d',
                 'category_2':'c',
                 'state_id':'c',
                 'subsector_id':'c',
                 'purchase_date_time_diff':'n',
                 'merchant_group_id':'c', 
                 'numerical_1':'n', 
                 'numerical_2':'n',
                 'most_recent_sales_range':'c', 
                 'most_recent_purchases_range':'c',
                 'avg_sales_lag3':'n', 
                 'avg_purchases_lag3':'n', 
                 'active_months_lag3':'n',
                 'avg_sales_lag6':'n', 
                 'avg_purchases_lag6':'n', 
                 'active_months_lag6':'n',
                 'avg_sales_lag12':'n', 
                 'avg_purchases_lag12':'n', 
                 'active_months_lag12':'n',
                 'category_4':'c',
                     'purchase_dow':'n',
                     'purchase_hour':'n'}

HT_FEATURE_WINDOWS = {'authorized_flag':10,
                      'card_id':10,
                      'city_id':10,
                      'category_1':10,
                      'installments':10,
                      'category_3':10,
                      'merchant_category_id':10,
                      'merchant_id':10,
                      'month_lag':10,
                      'purchase_amount':10,
                      'purchase_date':10,
                      'category_2':10,
                      'state_id':10,
                      'subsector_id':10,
                      'purchase_date_time_diff':10,
                     'merchant_group_id':10, 
                     'numerical_1':10, 
                     'numerical_2':10,
                     'most_recent_sales_range':10, 
                     'most_recent_purchases_range':10,
                     'avg_sales_lag3':10, 
                     'avg_purchases_lag3':10, 
                     'active_months_lag3':10,
                     'avg_sales_lag6':10, 
                     'avg_purchases_lag6':10, 
                     'active_months_lag6':10,
                     'avg_sales_lag12':10, 
                     'avg_purchases_lag12':10, 
                     'active_months_lag12':10,
                     'category_4':10,
                     'purchase_dow':10,
                     'purchase_hour':10}


NM_IMPUTE_DICT = {'category_3':'MISSING',
                  'merchant_id':'MISSING',
                  'category_2':'MISSING',
                  'most_recent_sales_range':'MISSING', 
                  'most_recent_purchases_range':'MISSING',
                  'avg_sales_lag3':-1, 
                  'avg_purchases_lag3':-1, 
                  'active_months_lag3':-1,
                  'avg_sales_lag6':-1, 
                  'avg_purchases_lag6':-1, 
                  'active_months_lag6':-1,
                  'avg_sales_lag12':-1, 
                  'avg_purchases_lag12':-1, 
                  'active_months_lag12':-1,
                  'category_4':'N'}

NM_FEATURE_TYPES = {'authorized_flag':'c',
                     'card_id':'k',
                     'city_id':'c',
                     'category_1':'c',
                     'installments':'n',
                     'category_3':'c',
                     'merchant_category_id':'c',
                     'merchant_id':'c',
                     'month_lag':'n',
                     'purchase_amount':'n',
                     'purchase_date':'d',
                     'category_2':'c',
                     'state_id':'c',
                     'subsector_id':'c',
                     'purchase_date_time_diff':'n',
                     'merchant_group_id':'c', 
                     'numerical_1':'n', 
                     'numerical_2':'n',
                     'most_recent_sales_range':'c', 
                     'most_recent_purchases_range':'c',
                     'avg_sales_lag3':'n', 
                     'avg_purchases_lag3':'n', 
                     'active_months_lag3':'n',
                     'avg_sales_lag6':'n', 
                     'avg_purchases_lag6':'n', 
                     'active_months_lag6':'n',
                     'avg_sales_lag12':'n', 
                     'avg_purchases_lag12':'n', 
                     'active_months_lag12':'n',
                     'category_4':'c',
                     'purchase_dow':'n',
                     'purchase_hour':'n'}

NM_FEATURE_WINDOWS = {'authorized_flag':10,
                      'card_id':10,
                      'city_id':10,
                      'category_1':10,
                      'installments':10,
                      'category_3':10,
                      'merchant_category_id':10,
                      'merchant_id':10,
                      'month_lag':10,
                      'purchase_amount':10,
                      'purchase_date':10,
                      'category_2':10,
                      'state_id':10,
                      'subsector_id':10,
                      'purchase_date_time_diff':10,
                      'merchant_group_id':10, 
                      'numerical_1':10, 
                      'numerical_2':10,
                      'most_recent_sales_range':10, 
                      'most_recent_purchases_range':10,
                      'avg_sales_lag3':10, 
                      'avg_purchases_lag3':10, 
                      'active_months_lag3':10,
                      'avg_sales_lag6':10, 
                      'avg_purchases_lag6':10, 
                      'active_months_lag6':10,
                      'avg_sales_lag12':10, 
                      'avg_purchases_lag12':10, 
                      'active_months_lag12':10,
                      'category_4':10,
                      'purchase_dow':10,
                      'purchase_hour':10}

HASH_TRICK_FEATURES = {'ht_hash_features':100,
                       'nm_hash_features':100}

L1_COLUMNS = ['authorized_flag','card_id','city_id','category_1',
              'installments','category_3','merchant_category_id','merchant_id',
              'month_lag','purchase_amount','purchase_date','category_2',
              'state_id','subsector_id','purchase_date_time_diff']

L2_COLUMNS = ['month_lag','purchase_amount','merchant_id','purchase_date','purchase_dow','purchase_hour']
L3_COLUMNS = ['month_lag','purchase_amount','merchant_id','purchase_date','purchase_dow','purchase_hour']
L4_COLUMNS = ['month_lag','purchase_amount','merchant_id','purchase_date','purchase_dow','purchase_hour']
L5_COLUMNS = ['month_lag','purchase_amount','merchant_id','purchase_date','purchase_dow','purchase_hour']
L6_COLUMNS = ['month_lag','purchase_amount','merchant_id','purchase_date','purchase_dow','purchase_hour']

def load_data_piece(data_id):
    full = pd.read_pickle('./data/full/splits/full_{}.pkl'.format(data_id))
    hist_trans = pd.read_pickle('./data/full/splits/hist_trans_{}.pkl'.format(data_id))
    new_merch_trans = pd.read_pickle('./data/full/splits/new_merch_trans_{}.pkl'.format(data_id))
    merch = pd.read_pickle('./data/full/splits/merch_{}.pkl'.format(data_id))
    
    full['first_active_month'] = np.where(full['first_active_month'].isna(),pd.to_datetime(datetime.date(2011,11,1)),full['first_active_month'] )
    full['first_active_month'] = pd.to_datetime(full['first_active_month'])
    
    for df in [hist_trans,new_merch_trans]:
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['purchase_dow'] = df['purchase_date'].dt.dayofweek
        df['purchase_hour'] = df['purchase_date'].dt.hour
        df['purchase_date_time_diff'] = ((datetime.date(2018,12,1) - df['purchase_date'].dt.date) / np.timedelta64(1, 'D')).astype(int)
    
    full['first_active_month_time_diff'] = ((datetime.date(2018,12,1) - full['first_active_month'].dt.date) / np.timedelta64(1, 'D')).astype(int)
    
    merch = merch.drop(['category_1','category_2','city_id','merchant_category_id','state_id','subsector_id'],axis=1)
    merch = merch.groupby('merchant_id').tail(1)
    hist_trans = hist_trans.merge(merch,how='left',on='merchant_id')
    new_merch_trans = new_merch_trans.merge(merch,how='left',on='merchant_id')
    
    return(full,hist_trans,new_merch_trans,merch)

def impute_values(full,hist_trans,new_merch_trans,merch):
    
    #Impute the values
    for key, values in HT_IMPUTE_DICT.items():
        hist_trans[key] = np.where(hist_trans[key].isna(),HT_IMPUTE_DICT[key],hist_trans[key])
        
    for key, values in NM_IMPUTE_DICT.items():
        new_merch_trans[key] = np.where(new_merch_trans[key].isna(),NM_IMPUTE_DICT[key],new_merch_trans[key])
        
    return(full,hist_trans,new_merch_trans,merch)
    
def gen_trans_features(trans_df,groupby_key,header,columns,seq,sort_key,
                       features_types_dict,feature_windows_dict):
    
    trans_features = gen_grouped_features(trans_df,
                                          groupby_key=groupby_key,
                                          sort_key=sort_key,
                                          feature_types_dict=features_types_dict,
                                          feature_windows_dict=feature_windows_dict,
                                          header=header,
                                          columns=columns,
                                          seq=seq)
    trans_features.reset_index(inplace=True)
    if seq:
        trans_features, trans_hashed_features = run_hash_trick(trans_features,'seq',header)
        trans_features = pd.concat([trans_features,trans_hashed_features],axis=1)
    trans_features = trans_features.fillna(value=0)
    return(trans_features)

def run_hash_trick(df,columns,table):

    replace_col_names = [cols for cols in df.columns if len(re.findall('_{}$'.format(columns),cols)) >0]
    df[replace_col_names] = df[replace_col_names].astype(str)
    hasher = FeatureHasher(n_features=HASH_TRICK_FEATURES['{}_hash_features'.format(table)],input_type="string")
    hashed_features = hasher.fit_transform(df[replace_col_names].values).todense()
    hashed_features = pd.DataFrame(hashed_features)
    hashed_features.columns = ['{}_seq_feat{}'.format(table,i) for i in range(HASH_TRICK_FEATURES['{}_hash_features'.format(table)])]
    
    df = df.drop(replace_col_names,axis=1)
    return(df,hashed_features)
    
def load_features(vl_list):
    
    vl_list = ['l1','l3','l4','l5','l6']
    
    def load_features_vl(vl):
        hist_trans_features = []
        new_merch_features = []
        for idx in range(32):
            hist_trans_features.append(pd.read_pickle('./data/full/features/{}/hist_trans_features_{}.pkl'.format(vl,idx)).set_index('card_id'))
            new_merch_features.append(pd.read_pickle('./data/full/features/{}/new_merch_features_{}.pkl'.format(vl,idx)).set_index('card_id'))
        return(pd.concat(hist_trans_features), pd.concat(new_merch_features))
    
    all_hist_trans_features = []
    all_new_merch_features = []
    for vl in vl_list:
        hist_trans_features_vl, new_merch_features_vl = load_features_vl(vl)
        
        if not(vl == 'l1'):
            ht_seq_cols = [cols for cols in hist_trans_features_vl.columns if len(re.findall('_seq',cols)) >0]
            hist_trans_features_vl.drop(ht_seq_cols,axis=1,inplace=True)
            nm_seq_cols = [cols for cols in new_merch_features_vl.columns if len(re.findall('_seq',cols)) >0]
            new_merch_features_vl.drop(nm_seq_cols,axis=1,inplace=True)
        all_hist_trans_features.append(hist_trans_features_vl)
        all_new_merch_features.append(new_merch_features_vl)
        
    return(pd.concat(all_hist_trans_features,axis=1), pd.concat(all_new_merch_features,axis=1))
    
if __name__ == '__main__':
    
#    for idx in range(32):
#        full,hist_trans,new_merch_trans,merch = load_data_piece(idx)
#        full,hist_trans,new_merch_trans,merch = impute_values(full,hist_trans,new_merch_trans,merch)
#        
#        hist_trans_features_l1 = gen_trans_features(trans_df=hist_trans,
#                                                    groupby_key=['card_id'],
#                                                    header='ht',
#                                                    columns=L1_COLUMNS,
#                                                    seq=True,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=HT_FEATURE_TYPES,
#                                                    feature_windows_dict=HT_FEATURE_WINDOWS)
#        
#        new_merch_features_l1 = gen_trans_features(trans_df=new_merch_trans,
#                                                    groupby_key=['card_id'],
#                                                    header='nm',
#                                                    columns=L1_COLUMNS,
#                                                    seq=True,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=NM_FEATURE_TYPES,
#                                                    feature_windows_dict=NM_FEATURE_WINDOWS)
#        hist_trans_features_l1.to_pickle('./data/full/features/l1/hist_trans_features_{}.pkl'.format(idx))
#        new_merch_features_l1.to_pickle('./data/full/features/l1/new_merch_features_{}.pkl'.format(idx))
#        
#        hist_trans_features_l3 = gen_trans_features(trans_df=hist_trans,
#                                                    groupby_key=['card_id','category_1'],
#                                                    header='ht_c1',
#                                                    columns=L3_COLUMNS,
#                                                    seq=False,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=HT_FEATURE_TYPES,
#                                                    feature_windows_dict=HT_FEATURE_WINDOWS)
#        
#        new_merch_features_l3 = gen_trans_features(trans_df=new_merch_trans,
#                                                    groupby_key=['card_id','category_1'],
#                                                    header='nm_c1',
#                                                    columns=L3_COLUMNS,
#                                                    seq=False,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=NM_FEATURE_TYPES,
#                                                    feature_windows_dict=NM_FEATURE_WINDOWS)
#        hist_trans_features_l3.to_pickle('./data/full/features/l3/hist_trans_features_{}.pkl'.format(idx))
#        new_merch_features_l3.to_pickle('./data/full/features/l3/new_merch_features_{}.pkl'.format(idx))
#        
#        hist_trans_features_l4 = gen_trans_features(trans_df=hist_trans,
#                                                    groupby_key=['card_id','category_2'],
#                                                    header='ht_c2',
#                                                    columns=L4_COLUMNS,
#                                                    seq=False,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=HT_FEATURE_TYPES,
#                                                    feature_windows_dict=HT_FEATURE_WINDOWS)
#        
#        new_merch_features_l4 = gen_trans_features(trans_df=new_merch_trans,
#                                                    groupby_key=['card_id','category_2'],
#                                                    header='nm_c2',
#                                                    columns=L3_COLUMNS,
#                                                    seq=False,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=NM_FEATURE_TYPES,
#                                                    feature_windows_dict=NM_FEATURE_WINDOWS)
#        hist_trans_features_l4.to_pickle('./data/full/features/l4/hist_trans_features_{}.pkl'.format(idx))
#        new_merch_features_l4.to_pickle('./data/full/features/l4/new_merch_features_{}.pkl'.format(idx))
#        
#        hist_trans_features_l5 = gen_trans_features(trans_df=hist_trans,
#                                                    groupby_key=['card_id','category_3'],
#                                                    header='ht_c3',
#                                                    columns=L5_COLUMNS,
#                                                    seq=False,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=HT_FEATURE_TYPES,
#                                                    feature_windows_dict=HT_FEATURE_WINDOWS)
#        
#        new_merch_features_l5 = gen_trans_features(trans_df=new_merch_trans,
#                                                    groupby_key=['card_id','category_3'],
#                                                    header='nm_c3',
#                                                    columns=L5_COLUMNS,
#                                                    seq=False,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=NM_FEATURE_TYPES,
#                                                    feature_windows_dict=NM_FEATURE_WINDOWS)
#        hist_trans_features_l5.to_pickle('./data/full/features/l5/hist_trans_features_{}.pkl'.format(idx))
#        new_merch_features_l5.to_pickle('./data/full/features/l5/new_merch_features_{}.pkl'.format(idx))
#        
#        hist_trans_features_l6 = gen_trans_features(trans_df=hist_trans,
#                                                    groupby_key=['card_id','category_4'],
#                                                    header='ht_c4',
#                                                    columns=L6_COLUMNS,
#                                                    seq=False,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=HT_FEATURE_TYPES,
#                                                    feature_windows_dict=HT_FEATURE_WINDOWS)
#        
#        new_merch_features_l6 = gen_trans_features(trans_df=new_merch_trans,
#                                                    groupby_key=['card_id','category_4'],
#                                                    header='nm_c4',
#                                                    columns=L6_COLUMNS,
#                                                    seq=False,
#                                                    sort_key='purchase_date',
#                                                    features_types_dict=NM_FEATURE_TYPES,
#                                                    feature_windows_dict=NM_FEATURE_WINDOWS)
#        hist_trans_features_l6.to_pickle('./data/full/features/l6/hist_trans_features_{}.pkl'.format(idx))
#        new_merch_features_l6.to_pickle('./data/full/features/l6/new_merch_features_{}.pkl'.format(idx))
    
    train = pd.read_pickle('./raw_data/train.pkl')
    test = pd.read_pickle('./raw_data/test.pkl')
    
    full = pd.concat([train,test])
    
    del train, test
    
    full['first_active_month'] = np.where(full['first_active_month'].isna(),pd.to_datetime(datetime.date(2011,11,1)),full['first_active_month'] )
    full['first_active_month'] = pd.to_datetime(full['first_active_month'])
    full['first_active_month_time_diff'] = ((datetime.date(2018,12,1) - full['first_active_month'].dt.date) / np.timedelta64(1, 'D')).astype(int)
    
    hist_trans_features, new_merch_features = load_features(['l1','l3','l4','l5','l6'])
    
    assert len(set(hist_trans_features.columns)) == len(hist_trans_features.columns)
    assert len(set(new_merch_features.columns)) == len(new_merch_features.columns)
    
    hist_trans_features = hist_trans_features.reset_index()
    new_merch_features = new_merch_features.reset_index()
    
    hist_trans_features.rename({'index':'card_id'},axis=1,inplace=True)
    new_merch_features.rename({'index':'card_id'},axis=1,inplace=True)
    
    full['feature_1'] = full['feature_1'].astype(str)
    full['feature_2'] = full['feature_2'].astype(str)
    full['feature_3'] = full['feature_3'].astype(str)
    dummie_vars_df = pd.get_dummies(full[['feature_1','feature_2','feature_3']],prefix_sep='_')
    full = pd.concat([full,dummie_vars_df],axis=1)
    dummie_vars = dummie_vars_df.columns
    
    del dummie_vars_df
    
    hist_trans_vars = list(set(hist_trans_features.columns) - set(['card_id']))
    new_merch_vars = list(set(new_merch_features.columns) - set(['card_id']))
    
    full = full.merge(hist_trans_features,how='left',on='card_id')
    full = full.merge(new_merch_features,how='left',on='card_id')
    
    del hist_trans_features, new_merch_features
    
    init_vars = ['first_active_month_time_diff']
    all_vars = [init_vars,dummie_vars,hist_trans_vars,new_merch_vars]
    PREDICTORS = list(itertools.chain.from_iterable(all_vars))
    TARGET = 'target'
    
    full[PREDICTORS] = full[PREDICTORS].fillna(0)
    
    train = full[~(full['target'].isna())]
    test = full[full['target'].isna()]
    
    xgb_params = {'max_depth': 6,
                  'eta': 0.1, 
                  'silent': 1,
                  'subsample':0.6,
                  'gamma':50,
                  'colsample_bytree':0.3,
                  'min_child_weight':50,
                  'objective':'reg:linear',
                  'booster':'gbtree',
                  'eval_metric':'rmse',
                  'seed':128}
    
    fit_params = {'num_boost_round':500}
    
    kf = KFold(n_splits=10)
    
    cv_scores_list = []
    feature_importance_list = []
    for train_index, valid_index in kf.split(train):
        dtrain = xgb.DMatrix(data=train[PREDICTORS].iloc[train_index],label=train[TARGET].iloc[train_index],feature_names=PREDICTORS)
        dvalid = xgb.DMatrix(data=train[PREDICTORS].iloc[valid_index],label=train[TARGET].iloc[valid_index],feature_names=PREDICTORS)
        
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        evals_result = {}
        xgb_cv = xgb.train(xgb_params, dtrain,**fit_params, evals=watchlist, evals_result=evals_result,verbose_eval=10)
        
        feature_importance = pd.DataFrame.from_dict(xgb_cv.get_fscore(),orient='index')
        feature_importance = feature_importance.reset_index()
        feature_importance.columns = ['variable','score']
        feature_importance_list.append(feature_importance)
        
        eval_scores = pd.DataFrame({'train_rmse':evals_result['train']['rmse'],
                                    'valid_rmse':evals_result['valid']['rmse']})
        cv_scores_list.append(eval_scores)
        
    cv_scores = pd.concat(cv_scores_list)
    cv_scores = cv_scores.reset_index()
    cv_scores = cv_scores.rename({'index':'iteration'},axis=1)
    cv_scores = cv_scores.groupby('iteration').apply(lambda x: pd.Series({'train_rmse_mean':np.mean(x['train_rmse']),
                                                                     'train_rmse_std':np.std(x['train_rmse']),
                                                                     'valid_rmse_mean':np.mean(x['valid_rmse']),
                                                                     'valid_rmse_std':np.std(x['valid_rmse'])}))
        
    feature_importance = pd.concat(feature_importance_list)
    feature_importance = feature_importance.groupby('variable').apply(lambda x: pd.Series({'count':len(x),
                                                                      'avg_score':np.mean(x['score']),
                                                                      'min_score':np.min(x['score']),
                                                                      'max_score':np.max(x['score'])}))
        
    keep_vars = feature_importance['variable'].values
    kf = KFold(n_splits=10)
    
    cv_scores_list = []
    feature_importance_list2 = []
    for train_index, valid_index in kf.split(train):
        dtrain = xgb.DMatrix(data=train[keep_vars].iloc[train_index],label=train[TARGET].iloc[train_index],feature_names=keep_vars)
        dvalid = xgb.DMatrix(data=train[keep_vars].iloc[valid_index],label=train[TARGET].iloc[valid_index],feature_names=keep_vars)
        
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        evals_result = {}
        xgb_cv = xgb.train(xgb_params, dtrain,**fit_params, evals=watchlist, evals_result=evals_result,verbose_eval=10)
        
        feature_importance = pd.DataFrame.from_dict(xgb_cv.get_fscore(),orient='index')
        feature_importance = feature_importance.reset_index()
        feature_importance.columns = ['variable','score']
        feature_importance_list2.append(feature_importance)
        
        eval_scores = pd.DataFrame({'train_rmse':evals_result['train']['rmse'],
                                    'valid_rmse':evals_result['valid']['rmse']})
        cv_scores_list.append(eval_scores)
        
    feature_importance2 = pd.concat(feature_importance_list2)
    feature_importance2 = feature_importance2.groupby('variable').apply(lambda x: pd.Series({'count':len(x),
                                                                      'avg_score':np.mean(x['score']),
                                                                      'min_score':np.min(x['score']),
                                                                      'max_score':np.max(x['score'])}))
    
        
    plt.figure(figsize=(16,8))
    plt.plot(cv_scores.index.values,cv_scores['train_rmse_mean'],color='r')
    plt.fill_between(cv_scores.index.values,cv_scores['train_rmse_mean']-cv_scores['train_rmse_std'],cv_scores['train_rmse_mean']+cv_scores['train_rmse_std'],alpha=0.1,color='r')
    plt.plot(cv_scores.index.values,cv_scores['valid_rmse_mean'],color='g')
    plt.fill_between(cv_scores.index.values,cv_scores['valid_rmse_mean']-cv_scores['valid_rmse_std'],cv_scores['valid_rmse_mean']+cv_scores['valid_rmse_std'],alpha=0.1,color='g')
    plt.axvline(xgb_cv.iloc[:,2].idxmin(),linestyle='--')
    plt.show()
    
    min_cv_score = cv_scores['valid_rmse_mean'].min()
    final_boost_rounds = cv_scores['valid_rmse_mean'].idxmin()
    
    final_fit_params = {'num_boost_round':final_boost_rounds}
    
    dtrain = xgb.DMatrix(data=train[keep_vars],label=train[TARGET],feature_names=keep_vars)
    dtest = xgb.DMatrix(data=test[keep_vars],feature_names=keep_vars)
    
    xgb_final = xgb.train(xgb_params, dtrain,**final_fit_params,verbose_eval=10)
    submission = test[['card_id']]
    submission['target'] = xgb_final.predict(dtest)
    
    ss = pd.read_pickle('./raw_data/sample_submission.pkl')
    ss = ss.drop(['target'],axis=1)
    
    ss = ss.merge(submission,how='left',on='card_id')
    ss.to_csv('./submissions/submission_{}.csv'.format(min_cv_score),index=False)
    
    

    
    
    