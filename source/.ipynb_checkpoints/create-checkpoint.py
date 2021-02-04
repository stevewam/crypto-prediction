import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from source.load import *

PROPERTIES = ['market_cap', 'price', 'volume', 'rank', 'market_share', 'age', 'roi']

def create_features(data, W):
    features = []

    for p in PROPERTIES:
        for w in range(1, W+1):
            col_name = "{}_lag_{}".format(p, w)
            data[col_name] = data.groupby(['sym'])[p].shift(w)
            features.append(col_name)

        data[p + '_mean'] = data.groupby(['sym'])[p].shift(1) \
                                    .transform(lambda x: x.rolling(W, min_periods=1).mean())
        data[p + '_std'] = data.groupby(['sym'])[p].shift(1) \
                                    .transform(lambda x: x.rolling(W, min_periods=1).std(ddof=0))
    
    data.dropna(inplace=True)
    
    return {'data': data, 'features': features}


def scale_col(df, base, col):
    mean = df[base + '_mean']
    std = df[base + '_std']
    std = np.where(std == 0, 0.001, std)
    
    return (df[col] - mean)/std


def normalize(df):
    df_scaled = df.copy()
    for p in PROPERTIES:
        df_scaled[p] = scale_col(df_scaled, p, p)
        
        cols = [col for col in df_scaled.columns if p+'_lag' in col]
        
        for col in cols:
            df_scaled[col] = scale_col(df_scaled, p, col)
    
    return df_scaled


def create_sets(df, features, target):
    criteria = {'train': "time < @VAL_START", 
                'val': "time >= @VAL_START & time < @TEST_START",
                'trainval': "time <= @TEST_START", 
                'test': "time >= @TEST_START"}
    
    scaled_df = normalize(df)

    sets = {}
    sets['features'] = features

    for key in criteria:
        sets[key] = {}
        sets[key]['ori'] = df.query(criteria[key])
        sets[key]['scaled'] = scaled_df.query(criteria[key])
        sets[key]['X'] = sets[key]['scaled'][features]
        sets[key]['Y'] = sets[key]['scaled'][target]
        
    return sets

def load_sets(data, target, W):
    features_dict = create_features(data, W)
    feat_df = features_dict['data']
    features = features_dict['features']
    
    sets = create_sets(feat_df, features, target)
    
    return sets
    