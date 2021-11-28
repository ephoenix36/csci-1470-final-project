import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer_utils import mark_as_return


class MeshedMemoryModel(tf.keras.Model):
    def __init__(self,):
        super(MeshedMemoryModel, self).__init__()
        
        # Note that the The input image is already broken images of objects within the image when passed in
        # TODO: pad inputs maybe???

        self.learning_rate = 0.01
        self.batch_size = 100
        self.hidden_size = 50

        ####################################################################
        ################### MEMORY AUGMENTED ATTENTION #####################
        ####################################################################
        # Pass each image in input through these layers
        self.wq = tf.keras.layers.Dense(self.hidden_size) # query
        self.wk = tf.keras.layers.Dense(self.hidden_size) # key
        self.wv = tf.keras.layers.Dense(self.hidden_size) # value
        # in the parper they initialized these with a uniform distribution???
        
        # These are learnable params that get appended onto the end of wk and wv to help allowing us to learn knowledge about how two objects may be related
        self.mk = tf.Variable(tf.random.normal([self.batch_size, self.hidden_size], stddev=.1)) 
        self.mv = self.mk = tf.Variable(tf.random.normal([self.batch_size, self.hidden_size], stddev=.1))
        
        # This is applied to wq*x, k, v
        #   k = [wk*x, mk]
        #   v = [wv*x, mv]
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
