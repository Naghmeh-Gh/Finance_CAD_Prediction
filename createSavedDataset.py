# -*- coding: utf-8 -*-

# Importing required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.train import Int64List,FloatList
from tensorflow.train import Feature,Features,Example
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Ensure working directory is the script's location

# Read the data, with its two pandas dataframe
data = pd.read_pickle('appml-assignment1-dataset-v2.pkl')
X = data['X'] # Features, including prices and datetime
y = data['y'] # CAD-high price

# Aligning the current CAD-close and the next-hour's CAD-high
cad_close = X['CAD-close'].values[:-1]
cad_high_next = y.values[1:]

# Computing fractional change
frac_change = (cad_high_next - cad_close) / cad_close # the fractional change

# Quantize into 22 bins
bin_edges = np.linspace(-0.001, 0.001, 21)
targets = np.digitize(frac_change, bins=bin_edges)
bin_edges

# Categorical attributes indicating the week day, month, and time in the dataframe X
X_valid = X.iloc[:-1].copy()
X_valid['weekday'] = X_valid['date'].dt.dayofweek
X_valid['hour'] = X_valid['date'].dt.hour
X_valid['month'] = X_valid['date'].dt.month

# Final transformation into tfrecords, and exporting as dataset.tfrecords
def _float_feature(value):
    return Feature(float_list=FloatList(value=value))
def _int64_feature(value):
    return Feature(int64_list=Int64List(value=[value]))

# Write TFRecord
with tf.io.TFRecordWriter('dataset.tfrecords') as f:
    for i in range(len(X_valid)):
        ticker_vals = X_valid.iloc[i].drop('date').drop(['weekday', 'hour', 'month']).values.astype(np.float32) # Droping the non-required symbols
        example = Example(features=Features(feature={
            'tickers': _float_feature(ticker_vals),
            'weekday': _int64_feature(X_valid.iloc[i]['weekday']),
            'hour': _int64_feature(X_valid.iloc[i]['hour']),
            'month': _int64_feature(X_valid.iloc[i]['month']),
            'target': _int64_feature(targets[i])
        }))
        f.write(example.SerializeToString())

print('\ndata is transformed to the dataset.tfrecords successfully!')