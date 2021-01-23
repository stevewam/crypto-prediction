import pandas as pd
import numpy as np

def rename_columns(df, suffix):
    if 'index' in df.columns:
        df.drop(columns='index', inplace=True)
    else:
        pass
    col_names = [col + '_' + suffix for col in df.columns]
    df.columns = col_names

def create_features(df, w):
    # Calculate mean
    mean_df = df.groupby('sym').shift(1).rolling(w).mean().reset_index()
    rename_columns(mean_df, '1_mean') # Numbers are assigned to the suffix for column ordering

    # Calculate median
    median_df = df.groupby('sym').shift(1).rolling(w).median().reset_index()
    rename_columns(median_df, '2_median')
    
    # Calculate standard deviation
    stdev_df = df.groupby('sym').shift(1).rolling(w).std().reset_index()
    rename_columns(stdev_df, '3_stdev')
    
    # Identify last value
    last_df = df.groupby('sym').shift(1)
    rename_columns(last_df, '4_last')
    
    # Identify first value and calculate difference between first and last value
    first_df = df.groupby('sym').shift(w)
    delta_df = pd.DataFrame()
    for col in first_df.columns:
        col_last = col + '_4_last'
        try:
            delta_df[col] = last_df[col_last] - first_df[col]
        except:
            pass
    rename_columns(delta_df, '5_delta')
    
    # Create base table to match created features with its respective time and coin
    base_df = df[['time', 'sym', 'price']]
    
    # Set current price as the target price to predict based on the features
    base_df = base_df.rename(columns={'price': 'target_price'})
    
    # Combine all features into a single table
    features_df = pd.concat([base_df, mean_df, median_df, stdev_df, last_df, delta_df], axis=1)
    
    # Clean up table by removing unnecessary columns and rearrange for ease of reference
    feature_cols = [col for col in features_df.columns]
    
    feature_cols.remove('age_1_mean')
    feature_cols.remove('age_2_median')
    feature_cols.remove('age_3_stdev')
    feature_cols.remove('age_5_delta')
    feature_cols.remove('name_4_last')
    feature_cols.remove('time_4_last')
    feature_cols.remove('time')
    feature_cols.remove('sym')
    feature_cols.remove('target_price')
    feature_cols.sort()

    features_df = features_df[['time', 'sym'] + feature_cols + ['target_price']]
    
    # Make time as index
    features_df['time'] = pd.to_datetime(features_df['time'])
    features_df = features_df.set_index('time', drop=True)
    
    # Remove points where there are insufficient data points to calculate features
    # For example, any time step that does not have w data points before it will have NA features
    features_df.dropna(inplace=True)
    
    return features_df