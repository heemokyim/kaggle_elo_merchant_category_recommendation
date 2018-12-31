# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:23:47 2018

@author: bcheung
"""

import pandas as pd
import numpy as np
import timeit


from multiprocessing import Pool

def parallelize_dataframe(df, func, split_key=None, num_partitions= 4, num_cores = 4):
    
#    df = hist_trans.copy()
#    func = partial_hist_trans
#    split_key='card_id'
#    start = timeit.timeit()
#    
    if split_key:
        df.sort_values([split_key],axis=0,inplace=True)
        unique, counts = np.unique(df[split_key].values, return_counts=True)
        cluster_values = np.minimum(np.floor(np.cumsum(counts)/(len(df)/4)),3)
        split_idx = np.argwhere(np.diff(cluster_values) == 1).reshape(-1)
        df_split = np.split(df,split_idx)
        
    else:
        df_split = np.array_split(df,num_partitions)

    pool = Pool(processes=num_cores)
    df = pd.concat(pool.map(func, df_split))
    
    pool.close()
    pool.join()
    return(df)