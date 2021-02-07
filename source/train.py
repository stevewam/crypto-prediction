"""Neural Net Custom Script

This script is train the custom neural net based on SKLearn
"""

from __future__ import print_function

import argparse
import os
import pandas as pd

import joblib

from sklearn.neural_network import MLPRegressor


def model_fn(model_dir):
    """Load model from the model_dir.

    Parameters
    ----------
    model_dir : str
        Location of model

    Returns
    -------
    object
        Loaded model object
    """
    print("Loading model.")

    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--hidden_layers', type=int, default=10)
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--random_state', type=int, default=100)

    args = parser.parse_args()

    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    hidden_layers = tuple([args.hidden_layers])

    model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                         max_iter=args.max_iter,
                         activation = 'relu',
                         solver='lbfgs',
                         early_stopping=True,
                         random_state=args.random_state)

    model.fit(train_x, train_y)

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))