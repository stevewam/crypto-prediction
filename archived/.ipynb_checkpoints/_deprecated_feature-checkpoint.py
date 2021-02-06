import pandas as pd
import numpy as np
import itertools
import ast

def rename_columns(df, suffix):
    if 'index' in df.columns:
        df.drop(columns='index', inplace=True)
    else:
        pass
    col_names = [col + '_' + suffix for col in df.columns]
    df.columns = col_names


def optimize_matrix(features_df, variable, show=True, cutoff=0.98):
    selected_features = [col for col in features_df.columns if variable in col if variable in col]
    df = features_df[selected_features]
    corr_matrix = df.corr().abs().round(2)
    lst = list(corr_matrix.columns)
    feature_combinations = []
#     selected_features = [col for col in corr_matrix.columns if variable in col if variable in col]
    for i in range(1,len(selected_features)):
        combinations = [list(x) for x in itertools.combinations(selected_features, i)]
        feature_combinations = feature_combinations + combinations
    
    selection_df = pd.DataFrame([str(comb) for comb in feature_combinations], columns=['features'])
    selection_df['feature_count'] = [len(comb) for comb in feature_combinations]
    selection_df['max_corr_count'] = selection_df['feature_count'] ** 2 - selection_df['feature_count']
    selection_df['corr_df'] = [(corr_matrix[corr_matrix.index.isin(comb)][comb] <= cutoff).values.sum() for comb in feature_combinations]

    selection_df['ratio'] = selection_df['corr_df']/selection_df['max_corr_count']
    selection_df['sum'] = [(corr_matrix[corr_matrix.index.isin(comb)][comb].values.sum()) for comb in feature_combinations]
    selection_df['exp'] = (selection_df['feature_count'] ** 2)/selection_df['sum']
#     selection_df.sort_values(by=['ratio'], ascending=False)
    if show:
        display(selection_df.nlargest(5, ['ratio', 'feature_count', 'exp']))   
    
    return ast.literal_eval(selection_df.nlargest(5, ['ratio', 'feature_count', 'exp']).iloc[0,0])

def create_features(df, w, target='price'):
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
    base_df = df[['time', 'sym', target]]
    
    # Set current price as the target price to predict based on the features
    base_df = base_df.rename(columns={target: 'target'})
    
    # Combine all features into a single table
    features_df = pd.concat([base_df, mean_df, median_df, stdev_df, last_df, delta_df], axis=1)
    
    
    feature_cols = []
    for prop in ['age', 'volume', 'roi', 'price', 'rank', 'market_cap']:
        selected_cols = optimize_matrix(features_df, prop, show=False)
#         print(selected_cols)
        feature_cols = feature_cols + selected_cols
        
    # Clean up table by removing unnecessary columns and rearrange for ease of reference
#     feature_cols = [col for col in features_df.columns]
    
#     feature_cols.remove('age_1_mean')
#     feature_cols.remove('age_2_median')
#     feature_cols.remove('age_3_stdev')
#     feature_cols.remove('age_5_delta')
#     feature_cols.remove('name_4_last')
#     feature_cols.remove('time_4_last')
#     feature_cols.remove('time')
#     feature_cols.remove('sym')
#     feature_cols.remove('target_price')
#     feature_cols.sort()
    print(feature_cols)
    features_df = features_df[['time', 'sym'] + feature_cols + ['target']]
    features_df['time'] = pd.to_datetime(features_df['time'])
    features_df = features_df.set_index('time', drop=True)
    
    # Remove points where there are insufficient data points to calculate features
    # For example, any time step that does not have w data points before it will have NA features
    features_df.dropna(inplace=True)
    
    return features_df