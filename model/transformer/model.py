import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer_utils import mark_as_return
from utils import *


class MeshedMemoryModel(tf.keras.Model):
    def __init__(self,):
        super(MeshedMemoryModel, self).__init__()
        
        # Note that the The input image is already broken images of objects within the image when passed in
        # TODO: pad inputs maybe???

        self.learning_rate = 0.01
        self.batch_size = 100
        self.hidden_size = 50

        self.attention = tf.keras.layers.MultiHeadAttention()
        
        
        ####################################################################
        ######################## ENCODING LAYER ############################
        ####################################################################
        self.t1 = tf.keras.layers.Dense(self.hidden_size, activation='relu') 
        self.t2 = tf.keras.layers.Dense(self.hidden_size)
        
        
        self.encoder = tf.keras.Sequential()
        # probably wrong...
        self.encoder.add(tf.keras.layers.Dense(100, activation='relu'))
        self.encoder.add(tf.keras.layers.Dense(100))

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    @tf.function
    def call(self):
        """
        :param ?
        :param ?
        :return: ?
        """
        pass

    def accuracy_function(self, prbs, labels, mask):
        """
        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(
            tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def loss_function(self, prbs, labels, mask):
        """
        Calculates the total model cross-entropy loss after one forward pass. 
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """
        return self.scce(labels, prbs, sample_weight=mask)
