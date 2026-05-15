
# Importing required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.train import Int64List,FloatList
from tensorflow.train import Feature,Features,Example
from tensorflow.keras.layers import Layer
import shutil
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Ensure working directory is the script's location
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from customImputerLayerDefinition import ImputerLayer # importing the imputer


#%% Define the parse function for the TFRecord file
num_tk_columns= 188
feature_description={'tickers':tf.io.FixedLenFeature([num_tk_columns],
                                  tf.float32,default_value=np.zeros(num_tk_columns)),
                      'weekday':tf.io.FixedLenFeature((),tf.int64,default_value=0),
                      'month':tf.io.FixedLenFeature((),tf.int64,default_value=0),
                      'hour':tf.io.FixedLenFeature((),tf.int64,default_value=0),
                      'target':tf.io.FixedLenFeature((),tf.int64,default_value=0)}

def parse_examples(serialized_examples):
    # Parses a serialized Example into feature dict + target
    examples=tf.io.parse_example(serialized_examples,feature_description)
    targets=examples['target']
    del examples["target"]
    return examples, targets

#%% Load and process the dataset (batches, parse_examples, Caches it in memory for faster access, shuffle for randomness)
dataset=tf.data.TFRecordDataset(['dataset.tfrecords']).batch(256).map(parse_examples).cache().shuffle(5000)
print('data is processed!\n')

#%% Split the dataset
datLen=dataset.reduce(0,lambda x,y: x+1)

n_valid=int(datLen.numpy()*.1) # 10% for validation
n_test=int(datLen.numpy()*.1) # 10% for testing
n_train=datLen.numpy()-n_valid-n_test # 80% for training

# keeping in cache for faster performance
dataset_tr=dataset.take(n_train).cache()
dataset_ts=dataset.skip(n_train).take(n_test).cache()
dataset_vd=dataset.skip(n_train+n_test).take(n_valid).cache()

#%% Create model inputs
inputDict={'tickers':tf.keras.Input(shape=(num_tk_columns,),dtype=tf.float32),
           'weekday':tf.keras.Input((),dtype=tf.int64),
           'month':tf.keras.Input((),dtype=tf.int64),
           'hour':tf.keras.Input((),dtype=tf.int64)}

#%% Apply Imputer, Normalizer to tickers
train_tickers_ds = dataset.take(n_train).map(lambda x, y: x['tickers'])

Imp = ImputerLayer()
Imp.adapt(train_tickers_ds)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train_tickers_ds.map(lambda x: Imp(x)))

#%% Embedding layers for categorical inputs
weekdayEmbs=tf.keras.layers.Embedding(8,3)(inputDict['weekday'])
monthEmbs=tf.keras.layers.Embedding(13,4)(inputDict['month'])
hourEmbs=tf.keras.layers.Embedding(25,20)(inputDict['hour'])

#%% Concatenate all processed inputs
preproced = tf.keras.layers.Concatenate(axis=-1)([
    normalizer(Imp(inputDict['tickers'])),
    weekdayEmbs,
    monthEmbs,
    hourEmbs
])

#%% Define the model structure
restMod=tf.keras.Sequential([
                            tf.keras.layers.Dense(256,activation='relu', kernel_initializer= 'he_normal'),
                            tf.keras.layers.Dropout(rate=0.2),

                            tf.keras.layers.Dense(256,activation='relu', kernel_initializer= 'he_normal'),
                            tf.keras.layers.Dropout(rate=0.2),

                            tf.keras.layers.Dense(22, activation='softmax')]) # 22 neurons for last layer, repres. 22 bins
decs=restMod(preproced)
model=tf.keras.Model(inputs=inputDict,outputs=decs)

#%% Compile the model
optimizer = tf.keras.optimizers.Adam(0.001) # Defining the learning rate for adams

loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
model.compile(loss=loss, optimizer=optimizer, metrics=metrics,)

#%% Train the model
history = model.fit(dataset_tr,epochs=50,verbose=2,validation_data=dataset_vd,callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)]) # Early stopping to prevent overfitting
print('training was processed!\n')

#%% Evaluate and save the model
model.evaluate(dataset_ts)
model.save('mySavedModel')

# convert it to zip,
shutil.make_archive('mySavedModel', 'zip', root_dir='.', base_dir='mySavedModel')

print('saving the model was processed!\n')
