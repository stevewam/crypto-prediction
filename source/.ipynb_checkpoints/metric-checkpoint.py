"""Metric Calculation and Trading Strategy

This script is used to evaluate the prediction results.

This file can also be imported as a module and contains the following
functions:

    * RMSE - calculate RMSE of the prediction results
    * update_mean - update previous mean based on new value
    * update_std - update previous standard deviation based on new value
    * trade - evaluate trading performance based on the prediction result and generate trading history
"""

import pandas as pd
import numpy as np


def RMSE(df, target):
    """Calculate RMSE of the prediction results

    Parameters
    ----------
    df : DataFrame
        DataFrame containing prediction results and actual target value
    target : str
        Column name of target value

    Returns
    -------
    float
        RMSE value
    """
    
    return np.sqrt(sum((df[target] - df['pred']) ** 2)/len(df[target]))


def update_mean(mean, t, new_value):
    """Update previous mean based on new value

    Parameters
    ----------
    mean : float
        Previous mean value
    t : int
        Number of samples including new value
    new_value : float
        Value of new sample

    Returns
    -------
    float
        New mean value
    """
    
    if t == 0:
        return new_value
    else:
        return (mean * (t - 1) + new_value) / t

    
def update_std(std, mean, new_mean, t, new_value):
    """Update previous standard deviation based on new value

    Parameters
    ----------
    std: float
        Previous standard deviation value
    mean : float
        Previous mean value
    new_mean : float
        New mean value
    t : int
        Number of samples including new value
    new_value : float
        Value of new sample

    Returns
    -------
    float
        New standard deviation value
    """
    
    if t == 0:
        return 0
    else:
        return np.sqrt((std ** 2 * (t - 1) + (new_value - new_mean) * (new_value - mean)) / t)

    
def trade(result, target, n=10, initial_value=10000):
    """Update previous standard deviation based on new value

    Parameters
    ----------
    result : DataFrame
        DataFrame containing prediction results and actual target value
    target : str
        Column name of target value
    n : int, optional
        Max. number of currency to invest in at each time step
    initial_value : float
        initial capital amount in USD
        
    Returns
    -------
    DataFrame
        DataFrame containing trading history including total returns, daily ROI, 
        currencies selected on each day, %returns and Sharpe ratio
    """
    
    total_value = initial_value
    mean_roi = 0
    std_roi = 0
    
    #Get all the date values to iterate over
    dates = list(set(result.index))
    dates.sort()
    
    history = {}
    
    #Calculate predicted ROI based on target type
    df = result.copy()
    if target == 'price':
        df['predicted_roi'] = (df['pred']/df['price_lag_1']) - 1
    else:
        df['predicted_roi'] = df['pred']
    df.sort_values(by='predicted_roi', ascending=False, inplace=True)
    
    #t is the number of time step, also represents the number of total samples
    t = 1

    for date in dates:
        #Filter to only entries on the related date and with positive predicted ROI
        temp_df = df.query('time == @date & predicted_roi > 0')

        if not temp_df.empty:
            #Get largest top n predicted ROI
            top_n = temp_df.nlargest(n, 'predicted_roi')
            
            #Get the value of n
            selected_n = len(top_n)
            
            #Get the list of currencies to invest in
            currencies = list(top_n.sym)
            
            #Calculate dollar value of returns on the date
            day_return = sum(top_n['roi'] * total_value / selected_n)
            
            #Calculate daily ROI
            day_roi = day_return/total_value
        else:
            #If there is no positive predicted ROI
            currencies = []
            day_return = 0
            day_roi = 0
        
        #Calculate percentage returns on the date
        total_value += day_return
        percent_returns = (total_value/initial_value - 1) * 100
        
        #Update mean and standard deviation
        prev_mean_roi = mean_roi
        mean_roi = update_mean(prev_mean_roi, t, day_roi)
        std_roi = update_std(std_roi, prev_mean_roi, mean_roi, t, day_roi)
        
        #Replace standard deviation with a small value in case =0 to prevent infinite sharpe ratio
        std_roi = 0.001 if (std_roi == 0) else std_roi
        
        #Calculate sharpe ratio based on updated mean and standard deviation
        sharpe_ratio = mean_roi/std_roi
        
        #Record in history
        history[date] = [total_value, day_roi, currencies, percent_returns, sharpe_ratio]

        t += 1
        
    print('Cumulative Returns: {:.2e}%, Sharpe Ratio : {:.4e}' \
                          .format(percent_returns, sharpe_ratio))
    
    #Convert history to DataFrame
    history = pd.DataFrame.from_dict(history, orient='index', 
                                     columns=['total', 'roi', 'currencies', '%return', 'sharpe'])
    
    return history