# Importing required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.train import Int64List,FloatList
from tensorflow.train import Feature,Features,Example
from tensorflow.keras.layers import Layer

class ImputerLayer(Layer):

      def __init__(self, **kwargs):
          super().__init__( **kwargs) # Initialize the custom layer

      def build(self,batch_input_shape):
          # Create a non-trainable weight called 'imps' to store the imputation value for each feature
          self.imps=self.add_weight(name='imps',shape=(batch_input_shape[-1],), # One imputation value per feature
          initializer="zeros",trainable=False) # Not to be updated via backpropagation
          super().build(batch_input_shape)

      def call(self, X):
          return tf.where(tf.math.is_nan(X),self.imps,X) # Replace NaNs in input X with the stored minimum values in 'imps'

      def adapt(self, dataset):
          # Ensure the layer is built by examining the shape of a sample batch
          for batch in dataset.take(1):
              self.build(batch.shape)

          # Initialize with +inf so tf.minimum can find the real minimums
          initial = tf.fill((self.imps.shape[0],), tf.constant(np.inf, dtype=tf.float32))

          def reduce_min(state, batch):
              # Replace NaNs with +inf so they do not affect the min computation
              cleaned = tf.where(tf.math.is_nan(batch), tf.fill(tf.shape(batch), np.inf), batch)
              return tf.minimum(state, tf.reduce_min(cleaned, axis=0)) # Compute minimum per feature across the batch

          mins = dataset.reduce(initial, reduce_min) # Aggregate minimums across all batches
          self.imps.assign(mins) # Assign the result to the internal variable

      def compute_output_shape(self,batch_input_shape):
          return batch_input_shape # Output shape is the same as input shape