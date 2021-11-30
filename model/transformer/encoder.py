import tensorflow as tf
from Attention import *

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention()
        self.