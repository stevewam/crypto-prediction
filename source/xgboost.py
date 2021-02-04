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

class xgboost:
    
    def __init__(self, prefix, data, W, target, hyperparams):
        self.session = sagemaker.Session()
        self.role = get_execution_role()
        self.bucket = self.session.default_bucket()
        self.prefix = prefix
        
        self.W = W
        self.target = target
        self.sets = create.load_sets(data, self.target, self.W)
        self.train_loc = os.path.join(DATA_DIR, 'train.csv')
        self.val_loc = os.path.join(DATA_DIR, 'val.csv')
        
        self.hyperparams = hyperparams
        
        self.model = None
        self.predictor = None
    
    
    def __del__(self):
        self.cleanup()
        
        
    def fit(self):
        pd.concat([self.sets['train']['Y'], self.sets['train']['X']], axis=1) \
                        .to_csv(self.train_loc, header=False, index=False)
        pd.concat([self.sets['val']['Y'], self.sets['val']['X']], axis=1) \
                        .to_csv(self.val_loc, header=False, index=False)

        train_location = self.session.upload_data(self.train_loc, key_prefix=self.prefix)
        val_location = self.session.upload_data(self.val_loc, key_prefix=self.prefix)

        train_input = sagemaker.s3_input(s3_data=train_location, content_type='text/csv')
        val_input = sagemaker.s3_input(s3_data=val_location, content_type='text/csv')

        container = get_image_uri(self.session.boto_region_name, 'xgboost')

        estimator = Estimator(container,
                              self.role, 
                              train_instance_count=1,
                              train_instance_type='ml.m4.xlarge',
                              output_path='s3://{}/{}/output'.format(self.bucket, self.prefix),
                              sagemaker_session=self.session)

        estimator.set_hyperparameters(**self.hyperparams)

        estimator.fit({'train': train_input, 'validation': val_input})
        
        self.model = estimator
    
    
    def init_predictor(self):
        self.predictor = self.model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
        self.predictor.content_type = 'text/csv'
        self.predictor.serializer = csv_serializer
        
        return self.predictor

    
    def predict(self, predict_set, unscaled=True, batch_size=250):
        X_val = self.sets[predict_set]['X'].values
        
        n_batch = int(round(len(X_val)/batch_size, -1))

        pred_Y = [self.predictor.predict(batch).decode('utf-8') for batch in np.array_split(X_val, n_batch)]
        pred_Y = [ast.literal_eval(batch) for batch in pred_Y]
        pred_Y = np.array([val for sublist in pred_Y for val in sublist])
        
        if unscaled:
            base = self.sets[predict_set]['ori'].copy()
            pred_mean = base[self.target+'_mean']
            pred_std = base[self.target+'_std']
            pred = (pred_Y * pred_std) + pred_mean
        else:
            base = self.sets[predict_set]['scaled'].copy()
            pred = pred_Y
    
        base['pred'] = pred
        base = base.set_index('time')

        return base
    
    
    def cleanup(self):
        self.predictor.delete_endpoint()
        os.remove(self.train_loc)
        os.remove(self.val_loc)
        
    