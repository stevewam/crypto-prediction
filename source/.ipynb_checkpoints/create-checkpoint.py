"""Feature Engineering and Data Splitting

This script is used to prepare the training, validation and test sets based on the 
data given. Features creation is combined with data splitting as the features depend
on the parameter w.

This file can also be imported as a module and contains the following
functions:

    * create_features - create features based on the underlying data
    * scale_col - scale dataframe column based on rolling mean and standard deviation
    * normalize - normalize all feature columns in the dataset
    * create_sets - generate a dictionary of different data sets in a variety of formats
                    required for training and visualization
    * load_sets - run routine functions to create features, normalize and split data into
                  different sets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from source.load import DATA_DIR, TRAIN_START, TRAIN_END, VAL_START, TEST_START

# Hardcoded feature columns base name
PROPERTIES = ['market_cap', 'price', 'volume', 'rank', 'market_share', 'age', 'roi']

def create_features(data, W):
    """Create features based on the underlying data.

    Parameters
    ----------
    data : DataFrame
        DataFrame object of the original dataset
    W : int
        The length of rolling window

    Returns
    -------
    dict
        A dictionary containing data which contains the created features
        and features which include column names of the created features
    """
    #List to contain newly created features
    features = []
    
    
    for p in PROPERTIES:
        
        #Create lag features
        for w in range(1, W+1):
            col_name = "{}_lag_{}".format(p, w)
            data[col_name] = data.groupby(['sym'])[p].shift(w)
            features.append(col_name)
        
        #Calculate rolling mean
        data[p + '_mean'] = data.groupby(['sym'])[p].shift(1) \
                                    .transform(lambda x: x.rolling(W, min_periods=1).mean())
        
        #Calculate rolling standard deviation
        data[p + '_std'] = data.groupby(['sym'])[p].shift(1) \
                                    .transform(lambda x: x.rolling(W, min_periods=1).std(ddof=0))
    
    data.dropna(inplace=True)
    
    return {'data': data, 'features': features}


def scale_col(df, base, col):
    """Scale dataframe column based on rolling mean and standard deviation

    Parameters
    ----------
    df : DataFrame
        DataFrame object of features set
    base : str
        Property name of the feature
    col : str
        Column name to be scaled

    Returns
    -------
    Series
        Series object containing the scaled features
    """
    #Find the column's respective rolling mean and standard deviation
    mean = df[base + '_mean']
    std = df[base + '_std']
    
    #If standard deviation is zero, replace it with a small number to prevent result = infinite 
    std = np.where(std == 0, 0.001, std)
    
    return (df[col] - mean)/std


def normalize(df):
    """Normalize all feature columns in the dataset

    Parameters
    ----------
    df : DataFrame
        DataFrame object of features set to be normalized

    Returns
    -------
    DataFrame
        DataFrame object with normalized features
    """
    df_scaled = df.copy()
    

    for p in PROPERTIES:
        #Scale the base column
        df_scaled[p] = scale_col(df_scaled, p, p)
        
        #Find all columns of lag features
        cols = [col for col in df_scaled.columns if p+'_lag' in col]
        
        #Scale all lag features
        for col in cols:
            df_scaled[col] = scale_col(df_scaled, p, col)
    
    return df_scaled


def create_sets(df, features, target):
    """Generate a dictionary of different data sets in a variety of formats
    required for training and visualization

    Parameters
    ----------
    df : DataFrame
        DataFrame object of features set
    features: list
        List of feature column names
    target: str
        Column name of target feature to be fitted
        
    Returns
    -------
    dict
        Dictionary contains 4 data sets which include train, val, trainval
        and test. Each set has different variations including ori (the original feature set),
        scaled (scaled feature set), X (contains features only), Y (contains target only)
    """
    #Hardcoded criteria 
    criteria = {'train': "time < @VAL_START", 
                'val': "time >= @VAL_START & time < @TEST_START",
                'trainval': "time <= @TEST_START", 
                'test': "time >= @TEST_START"}
    
    #Use normalize function to normalize all lag features
    scaled_df = normalize(df)
    
    sets = {}
    sets['features'] = features

    for key in criteria:
        sets[key] = {}
        #Store original feature set
        sets[key]['ori'] = df.query(criteria[key])
        #Store scaled feature set
        sets[key]['scaled'] = scaled_df.query(criteria[key])
        #Store feature columns
        sets[key]['X'] = sets[key]['scaled'][features]
        #Store target column
        sets[key]['Y'] = sets[key]['scaled'][target]
        
    return sets

def load_sets(data, target, W):
    """Run routine functions to create features, normalize and split data into
    different sets

    Parameters
    ----------
    data : DataFrame
        DataFrame object of the original dataset
    target: str
        Column name of target feature to be fitted
    W : int
        The length of rolling window

    Returns
    -------
    dict
        Dictionary contains 4 data sets which include train, val, trainval
        and test. Each set has different variations including ori (the full dataset),
        scaled (scaled dataset), X (contains features only), Y (contains target only)
    """
    
    #Create features
    features_dict = create_features(data, W)
    #Get features set
    feat_df = features_dict['data']
    #Get feature column names
    features = features_dict['features']
    
    #Create sets
    sets = create_sets(feat_df, features, target)
    
    return sets
    