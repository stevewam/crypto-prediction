"""Data Loading

This script is used to load the data and calculate properties.

This file can also be imported as a module and contains the following
functions:

    * read_data - read cryptocurrency history data
    * process_data - calculate cryptocurrency properties based on the raw data
    * filter data - filter data to only include currency with higher than $1,000,000 daily average market cap
                    and higher than 10,000 average daily volume
    * load_data - run data loading routine of reading, filtering and processing

"""

import pandas as pd
import os

DATA_DIR = './data'
DATA_FILE = 'crypto-historical-data.csv'

#Predetermined, hardcoded dates to split train, validation and test sets
VAL_START = pd.Timestamp('2016-04-25')
TEST_START = pd.Timestamp('2017-04-25')

TRAIN_START = pd.Timestamp('2016-04-28')
TRAIN_END = pd.Timestamp('2017-04-24')


def read_data(file_location):
    """Read cryptocurrency history data file

    Parameters
    ----------
    file_location : str
        Location of cryptocurrency history data file

    Returns
    -------
    DataFrame
        History data in DataFrame object sorted by symbol and time
    """
    data = pd.read_csv(file_location, 
                   parse_dates=['time'], 
                   index_col=0, 
                   keep_default_na=False,
                   header=0,
                   names=['market_cap', 'name', 'price', 'sym', 'time', 'volume'])
    
    data.sort_values(by=['sym', 'time'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def process_data(data):
    """Calculate cryptocurrency properties based on the raw data

    Parameters
    ----------
    data : DataFrame
        DataFrame loaded by read_data

    Returns
    -------
    DataFrame
        Modified DataFrame with new properties included
    """
    #Calculate rank based on market cap
    data['rank'] = data.groupby("time")["market_cap"] \
                    .rank("dense", ascending=False) \
                    .astype(int)
    
    #Calculate market share by dividing market cap with the sum of all market caps
    data['market_share'] = data.groupby('time')["market_cap"] \
                        .apply(lambda x: x/float(x.sum()))
    
    #Determine age since the first time the currency appears in the data
    data['age'] = data.groupby(['sym'])["time"] \
                        .apply(lambda x: x - min(x)) \
                        .dt.days + 1
    
    #Calculate ROI based on previous price
    previous_price = data.groupby(['sym'])['price'].shift(-1)
    data['roi'] = data['price']/previous_price - 1
    
    return data


def filter_data(data):
    """Filter data to only include currency with higher than $1,000,000 daily average market cap
    
    Parameters
    ----------
    data : DataFrame
        DataFrame loaded by process_data

    Returns
    -------
    DataFrame
        Filtered data based on market cap and volume
    """
    
    #Exclude validation and test set to determine average market cap and daily volume
    window = data.query('time < @VAL_START')
    
    #Calculate average market cap and daily volume
    mean = window.groupby('sym').mean()
    
    #Find currencies with > $1,000,000 market cap and > 10,000 volume
    symbols = mean.query("market_cap > 1000000 & volume > 10000") \
                        .index.unique()
    
    #Filter entire dataset: train, val and test
    filtered = data.query('sym in @symbols')
    
    return {'data': filtered, 'symbols': symbols}


def load_data():
    """Run data loading routine of reading, filtering and processing
    
    Returns
    -------
    DataFrame
        Data with new properties and filtered
    """
    raw = read_data(os.path.join(DATA_DIR, DATA_FILE))
    data = process_data(raw)
    
    #Returned filtered data
    return filter_data(data)