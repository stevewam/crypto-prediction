import numpy as np
import pandas as pd
import boto3
import ast
import json
import os

import pkg_resources
pkg_resources.require("sagemaker==1.72.0")
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer
from sagemaker.estimator import Estimator
from sagemaker.sklearn.estimator import SKLearn

from source.load import DATA_DIR
from source import create

class deepar:
    
    def __init__(self, prefix, data, W, target, hyperparams):
        self.session = sagemaker.Session()
        self.role = get_execution_role()
        self.bucket = self.session.default_bucket()
        self.prefix = prefix
        
        self.W = W
        self.target = target
        self.sets = create.load_sets(data, self.target, self.W)
        self.features = self.sets['features']
        self.trainval_loc = os.path.join(DATA_DIR, 'trainval.json')
        
        self.hyperparams = hyperparams
        self.context_length = hyperparams['context_length']
        
        self.model = None
        self.predictor = None
        self.mapping = {}
        self.pred = pd.DataFrame()
    
    
    def __del__(self):
        self.cleanup()


    def write_json_dataset(self, features_df, filename): 
        symbols = features_df['sym'].unique()
        with open(filename, 'wb') as f:

            for idx, sym in enumerate(symbols):
                sym_df = features_df[features_df['sym'] == sym]
                sym_df = sym_df.drop(columns='sym')

                json_obj = {"start": str(sym_df['time'].iloc[0]), 
                            "target": list(sym_df[self.target]), 
                            "cat":[idx], 
                            "dynamic_feat":[list(sym_df[column]) for column in self.features]}

                json_line = json.dumps(json_obj) + '\n'
                json_line = json_line.encode('utf-8')

                f.write(json_line)

                self.mapping[sym] = idx
        print('JSON file created at ' + filename)
        
        
    def fit(self):
        self.write_json_dataset(self.sets['trainval']['scaled'], self.trainval_loc)

        trainval_location = self.session.upload_data(self.trainval_loc, key_prefix=self.prefix)

        container = get_image_uri(self.session.boto_region_name,'forecasting-deepar')

        estimator = Estimator(container,
                              role=self.role,   
                              train_instance_count=1, 
                              train_instance_type='ml.m4.xlarge', 
                              output_path='s3://{}/{}/output'.format(self.bucket, self.prefix),
                              sagemaker_session=self.session)


        estimator.set_hyperparameters(**self.hyperparams)

        estimator.fit({'train': trainval_location})
        
        self.model = estimator
    
    
    def init_predictor(self):
        self.predictor = self.model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
        
        return self.predictor
    
    
    def json_predictor_input(self, features_df, date, num_samples=50):
        instances = []
        symbols = features_df['sym'].unique()
        look_back_date = date - pd.Timedelta(self.context_length, 'D')
        window = features_df.query("time >= @look_back_date & time <= @date")

        for sym in symbols:
            idx = self.mapping[sym]
            sym_window = window.query("sym == @sym")

            if sym_window.empty:
                continue

            json_obj = {"start": str(list(sym_window['time'])[0]), 
                        "target": list(sym_window[self.target])[:-1],
                        "cat":[idx], 
                        "dynamic_feat":[list(sym_window[column]) for column in self.features]}
            instances.append(json_obj)

        configuration = {"num_samples": num_samples, 
                         "output_types": ["mean"]}

        request_data = {"instances": instances, 
                        "configuration": configuration}

        json_request = json.dumps(request_data).encode('utf-8')

        return json_request

    
    def decode_prediction(self, prediction, encoding='utf-8'):
        '''Accepts a JSON prediction and returns a list of prediction data.
        '''
        prediction_data = json.loads(prediction.decode(encoding))
        prediction_list = []
        
        for k in range(len(prediction_data['predictions'])):
            prediction_list.append(pd.DataFrame(data=prediction_data['predictions'][k]['mean']))
        
        return prediction_list

    
    def loop_predict(self, features_df, start, end):
        dates = list(set(features_df[(features_df.time >= start) & (features_df.time <= end)]['time']))

        df = pd.DataFrame([])
        for date in dates:
            test_features = self.json_predictor_input(features_df, date)
            json_prediction = self.predictor.predict(test_features)
            pred = [float(x.values.squeeze()) for x in self.decode_prediction(json_prediction)]
            temp_df = pd.DataFrame(zip([sym for sym in features_df['sym'].unique()], pred), 
                                   columns=['sym', 'pred'])
            temp_df['time'] = date

            df = df.append(temp_df, ignore_index=True)

        return df

    
    def predict(self, predict_set, unscaled=True, recalculate=True):
        trainval = self.sets['trainval']['scaled']
        test = self.sets['test']['scaled']
        X_val = pd.concat([trainval, test], axis=0)
        
        if predict_set == 'test':
            start = TEST_START
            end = X_val.time.max()
        elif predict_set == 'val':
            start = VAL_START
            end = TEST_START - pd.Timedelta(1, 'D')
        else:
            start = TRAIN_START - pd.Timedelta(self.context_length, 'D')
            end = TRAIN_END
            
        if self.pred.empty or recalculate:
            self.pred = self.loop_predict(X_val, start, end)
            
        pred_Y = self.pred
            
        if unscaled:
            base = self.sets[predict_set]['ori'].copy()
            base = pd.merge(base, pred_Y, how='left', left_on=['time', 'sym'], right_on=['time', 'sym'])
            pred_mean = base[self.target+'_mean'].reset_index(drop=True)
            pred_std = base[self.target+'_std'].reset_index(drop=True)
            base['pred'] = (base['pred'] * pred_std) + pred_mean
        else:
            base = self.sets[predict_set]['scaled'].copy()
            base = pd.merge(base, pred_Y, how='left', left_on=['time', 'sym'], right_on=['time', 'sym'])
            
        base = base.set_index('time')

        return base
    
    
    def cleanup(self):
        self.predictor.delete_endpoint()
        os.remove(self.trainval_loc)
    
        
    