# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:23:47 2018

@author: bcheung
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool

def parallelize_dataframe(df, func, split_key=None, num_partitions= 4, num_cores = 4):
    if split_key:
        split_rows = int(len(df) / 4)
        
        key_maps = df.groupby(split_key)[split_key].agg('count').cumsum()/split_rows
        key_maps = np.floor(key_maps).astype(int).to_dict()
        clust_map = df['card_id'].map(key_maps)
        split_idx = np.where(~(clust_map == clust_map.shift(1)))[0]
        df_split = np.split(df,split_idx)
        
    else:
        df_split = np.array_split(df,num_partitions)
    
    pool = Pool(processes=num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return(df)