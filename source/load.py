import pandas as pd
import os

DATA_DIR = './data'
DATA_FILE = 'crypto-historical-data.csv'

VAL_START = pd.Timestamp('2016-04-25')
TEST_START = pd.Timestamp('2017-04-25')

TRAIN_START = pd.Timestamp('2016-04-28')
TRAIN_END = pd.Timestamp('2017-04-24')

def read_data(file_location):
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
    data['rank'] = data.groupby("time")["market_cap"] \
                    .rank("dense", ascending=False) \
                    .astype(int)

    data['market_share'] = data.groupby('time')["market_cap"] \
                        .apply(lambda x: x/float(x.sum()))

    data['age'] = data.groupby(['sym'])["time"] \
                        .apply(lambda x: x - min(x)) \
                        .dt.days + 1

    previous_price = data.groupby(['sym'])['price'].shift(-1)
    data['roi'] = data['price']/previous_price - 1
    return data

def filter_data(data):
    window = data.query('time < @VAL_START')
    
    mean = window.groupby('sym').mean()
    symbols = mean.query("market_cap > 1000000 & volume > 10000") \
                        .index.unique()
    
    filtered = data.query('sym in @symbols')
    
    return {'data': filtered, 'symbols': symbols}

def load_data():
    raw = read_data(os.path.join(DATA_DIR, DATA_FILE))
    data = process_data(raw)
    
    return filter_data(data)