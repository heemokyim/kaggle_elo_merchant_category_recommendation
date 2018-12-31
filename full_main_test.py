# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:06:52 2018

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

RECREATE = True

train = pd.read_pickle('./data/sample/train_sample.pkl')
hist_trans = pd.read_pickle('./data/sample/historical_transactions_sample.pkl')
new_merch_trans = pd.read_pickle('./data/sample/new_merchant_transactions_sample.pkl')
merch = pd.read_pickle('./data/sample/merchants_sample.pkl')

HT_IMPUTE_DICT = {'category_3':'MISSING',
                  'merchant_id':'MISSING',
                  'category_2':'MISSING'}

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
                 'purchase_date_time_diff':'n'}

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
                 'purchase_date_time_diff':10}

NM_IMPUTE_DICT = {'category_3':'MISSING',
                  'merchant_id':'MISSING',
                  'category_2':'MISSING'}

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
                     'purchase_date_time_diff':'n'}

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
                 'purchase_date_time_diff':10}

HASH_TRICK_FEATURES = {'ht_hash_features':100,
                       'nm_hash_features':100}

def run_hash_trick(df,columns,table):

    replace_col_names = [cols for cols in df.columns if len(re.findall('_{}$'.format(columns),cols)) >0]
    df[replace_col_names] = df[replace_col_names].astype(str)
    hasher = FeatureHasher(n_features=HASH_TRICK_FEATURES['{}_hash_features'.format(table)],input_type="string")
    hashed_features = hasher.fit_transform(df[replace_col_names].values).todense()
    hashed_features = pd.DataFrame(hashed_features)
    hashed_features.columns = ['{}_seq_feat{}'.format(table,i) for i in range(HASH_TRICK_FEATURES['{}_hash_features'.format(table)])]
    
    df = df.drop(replace_col_names,axis=1)
    return(df,hashed_features)

#Prep the data
train['first_active_month'] = np.where(train['first_active_month'].isna(),pd.to_datetime(datetime.date(2011,11,1)),train['first_active_month'] )


train['first_active_month'] = pd.to_datetime(train['first_active_month'])

for df in [hist_trans,new_merch_trans]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['purchase_dow'] = df['purchase_date'].dt.dayofweek
    df['purchase_hour'] = df['purchase_date'].dt.hour
    df['purchase_date_time_diff'] = ((datetime.date(2018,12,1) - df['purchase_date'].dt.date) / np.timedelta64(1, 'D')).astype(int)

#Create some new features
train['first_active_month_time_diff'] = ((datetime.date(2018,12,1) - train['first_active_month'].dt.date) / np.timedelta64(1, 'D')).astype(int)

#Impute the values
for key, values in HT_IMPUTE_DICT.items():
    hist_trans[key] = np.where(hist_trans[key].isna(),HT_IMPUTE_DICT[key],hist_trans[key])
    
for key, values in NM_IMPUTE_DICT.items():
    new_merch_trans[key] = np.where(new_merch_trans[key].isna(),NM_IMPUTE_DICT[key],new_merch_trans[key])
    


if RECREATE:
    #Run a parallel process to extract the features from the historical transactions table
    partial_hist_trans = partial(gen_grouped_features,
                                 groupby_key='card_id',
                                 sort_key='purchase_date',
                                 feature_types_dict=HT_FEATURE_TYPES,
                                 feature_windows_dict=HT_FEATURE_WINDOWS,
                                 header='ht')
    
    hist_trans_features = parallelize_dataframe(hist_trans,partial_hist_trans,split_key='card_id')
    hist_trans_features.reset_index(inplace=True)
    hist_trans_features, hist_trans_hashed_features = run_hash_trick(hist_trans_features,'seq','ht')
    hist_trans_features = pd.concat([hist_trans_features,hist_trans_hashed_features],axis=1)
    hist_trans_features = hist_trans_features.fillna(value=0)
    hist_trans_features.to_pickle('./data/sample/hist_trans_features.pkl')
    
    #Run a parallel process to extract the features from the new merchants table
    partial_new_merch_trans = partial(gen_grouped_features,
                                 groupby_key='card_id',
                                 sort_key='purchase_date',
                                 feature_types_dict=NM_FEATURE_TYPES,
                                 feature_windows_dict=NM_FEATURE_WINDOWS,
                                 header='nm')
    
    new_merch_features = parallelize_dataframe(new_merch_trans,partial_new_merch_trans,split_key='card_id')
    new_merch_features.reset_index(inplace=True)
    new_merch_features, new_merch_hashed_features = run_hash_trick(new_merch_features,'seq','nm')
    new_merch_features = pd.concat([new_merch_features,new_merch_hashed_features],axis=1)
    new_merch_features = new_merch_features.fillna(value=0)
    new_merch_features.to_pickle('./data/sample/new_merch_features.pkl')
    
else:
    hist_trans_features = pd.read_pickle('./data/sample/hist_trans_features.pkl')
    new_merch_features = pd.read_pickle('./data/sample/new_merch_features.pkl')

hist_trans_feats = list(set(hist_trans_features.columns) - set(['card_id']))
new_merch_feats = list(set(new_merch_features.columns) - set(['card_id']))

train['feature_1'] = train['feature_1'].astype(str)
train['feature_2'] = train['feature_2'].astype(str)
train['feature_3'] = train['feature_3'].astype(str)
dummie_vars_df = pd.get_dummies(train[['feature_1','feature_2','feature_3']],prefix_sep='_')
train = pd.concat([train,dummie_vars_df],axis=1)
dummie_vars = dummie_vars_df.columns

train = train.merge(hist_trans_features,how='left',on='card_id')
train = train.merge(new_merch_features,how='left',on='card_id')

init_vars = ['first_active_month_time_diff']
all_vars = [init_vars,dummie_vars,hist_trans_feats,new_merch_feats]
PREDICTORS = list(itertools.chain.from_iterable(all_vars))
TARGET = 'target'

<<<<<<< Updated upstream:feature_gen.py
full[PREDICTORS] = full[PREDICTORS].fillna(value=0)

xgb_params = {'max_depth': [3,6,9,12,15,18,21], 
              'subsample':[0.2,0.4,0.6,0.8],
              'gamma':[0,0.1,1,10,100,1000],
              'colsample_bytree':[0.2,0.4,0.6,0.8],
              'min_child_weight':[0,0.1,1,10,100,1000]}
=======
xgb_params = {'max_depth': 6, 
              'eta': 0.1, 
              'silent': 1,
              'subsample':0.6,
              'gamma':50,
              'colsample_bytree':0.3,
              'min_child_weight':100,
              'objective':'reg:linear',
              'booster':'gbtree',
              'eval_metric':'rmse',
              'seed':128}
>>>>>>> Stashed changes:full_main_test.py

fit_params = {'num_boost_round':250}

dtrain = xgb.DMatrix(data=train[PREDICTORS],
                     label=train[TARGET],feature_names=PREDICTORS)
xgb_cv = xgb.cv(xgb_params, dtrain,**fit_params,nfold=10,seed=128,verbose_eval=False)

plt.figure(figsize=(16,8))
plt.plot(xgb_cv.index.values,xgb_cv.iloc[:,0],color='r')
plt.fill_between(xgb_cv.index.values,xgb_cv.iloc[:,0]-xgb_cv.iloc[:,1],xgb_cv.iloc[:,0]+xgb_cv.iloc[:,1],alpha=0.1,color='r')
plt.plot(xgb_cv.index.values,xgb_cv.iloc[:,2],color='g')
plt.fill_between(xgb_cv.index.values,xgb_cv.iloc[:,2]-xgb_cv.iloc[:,3],xgb_cv.iloc[:,2]+xgb_cv.iloc[:,3],alpha=0.1,color='g')
plt.axvline(xgb_cv.iloc[:,2].idxmin(),linestyle='--')
plt.show()

best_xgb_score = xgb_cv.iloc[:,2].min()

xgb_final = xgb.train(xgb_params, dtrain,**fit_params,verbose_eval=False)

feature_importance = pd.DataFrame.from_dict(xgb_final.get_fscore(),orient='index')
feature_importance = feature_importance.reset_index()
feature_importance.columns = ['variable','score']

keep_vars = feature_importance[feature_importance['score']>10]['variable']
dtrain = xgb.DMatrix(data=train[keep_vars],label=train[TARGET],feature_names=keep_vars)
xgb_cv = xgb.cv(xgb_params, dtrain,**fit_params,nfold=10,seed=128,verbose_eval=False)


#3.705
"""
Random Search
"""

xgb_params_search = {'max_depth': [3,6,9,12,15,18,21], 
              'subsample':[0.2,0.4,0.6,0.8],
              'gamma':[0,0.1,1,10,100,1000],
              'colsample_bytree':[0.2,0.4,0.6,0.8],
              'min_child_weight':[0,0.1,1,10,100,1000]}

xgb_regressor = xgb.XGBRegressor(learning_rate=0.1, n_estimators=600, objective='reg:linear',silent=True,booster='gbtree')
folds = 3
param_comb = 200

skf = KFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb_regressor, 
                                   param_distributions=xgb_params_search, 
                                   n_iter=param_comb, 
                                   scoring='mean_squared_error', 
                                   n_jobs=4, 
                                   cv=skf.split(full[PREDICTORS].values,full[TARGET].values), 
                                   verbose=3, 
                                   random_state=1001)

random_search.fit(full[PREDICTORS].values,full[TARGET].values)










