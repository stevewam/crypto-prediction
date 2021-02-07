import pandas as pd
import numpy as np

def RMSE(df, target):
    return np.sqrt(sum((df[target] - df['pred']) ** 2)/len(result[target]))

def update_mean(mean, t, new_value):
    if t == 0:
        return new_value
    else:
        return (mean * (t - 1) + new_value) / t

# Function to update standard deviation based on new value
def update_std(std, mean, new_mean, t, new_value):
    if t == 0:
        return 0
    else:
        return np.sqrt((std ** 2 * (t - 1) + (new_value - new_mean) * (new_value - mean)) / t)

# Function to execute the trading strategy over the trading horizon using asset matrix
def evaluate(result, target, n=10, initial_value=10000):
    total_value = initial_value
    mean_roi = 0
    std_roi = 0

    dates = list(set(result.index))
    dates.sort()
    
    history = {}
    
    df = result.copy()
    if target == 'price':
        df['predicted_roi'] = (df['pred']/df['price_lag_1']) - 1
    else:
        df['predicted_roi'] = df['pred']
    df.sort_values(by='predicted_roi', ascending=False, inplace=True)
    
    t = 1

    for date in dates:
        temp_df = df.query('time == @date & predicted_roi > 0')

        if not temp_df.empty:
            top_n = temp_df.nlargest(n, 'predicted_roi')
            selected_n = len(top_n)
            currencies = list(top_n.sym)
            day_return = sum(top_n['roi'] * total_value / selected_n)
            day_roi = day_return/total_value
        else:
            currencies = []
            day_return = 0
            day_roi = 0
        
        total_value += day_return
        percent_returns = (total_value/initial_value - 1) * 100

        prev_mean_roi = mean_roi
        mean_roi = update_mean(prev_mean_roi, t, day_roi)
        std_roi = update_std(std_roi, prev_mean_roi, mean_roi, t, day_roi)
        std_roi = 0.001 if (std_roi == 0) else std_roi
        sharpe_ratio = mean_roi/std_roi
        
        history[date] = [total_value, day_roi, currencies, percent_returns, sharpe_ratio]

        t += 1
        
    print('Cumulative Returns: {:.2e}%, Sharpe Ratio : {:.4e}' \
                          .format(percent_returns, sharpe_ratio))
    
    history = pd.DataFrame.from_dict(history, orient='index', 
                                     columns=['total', 'roi', 'currencies', '%return', 'sharpe'])
    
    return history