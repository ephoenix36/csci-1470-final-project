import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from attention import *
from utils import *

class EncoderLayer(Layer):
    def __init__(self, output_size, key_size=64, value_size=64, hidden_size=1024, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.attention = MultiHeadAttention(output_size, key_size, value_size)
        
        self.ff = PositionWiseFeedForward(output_size, hidden_size, dropout)
    
    @tf.function
    def call(self, queries, keys, values):
        output = self.attention(queries, keys, values)
        output = self.ff(output)
        return output
    
class MultiLevelEncoder(Layer):
    # TODO: change correct output_size to correct value
    def __init__(self, num_layers, padding_index, output_size=512, hidden_size=1024, dropout=0.1):
        super(MultiLevelEncoder, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.layers = [EncoderLayer(output_size, hidden_size, dropout) for _ in range(num_layers)]
        
        self.padding_index = padding_index
    
    @tf.function
    def call(self, input):
        attention_mask = (input == self.padding_index)
        # in paper they unsqueeze this mask to (batch, 1, 1, seq_len)
        
        outputs = []
        output = input
        for layer in self.layers:
            output = layer(output, output, output, attention_mask)
            outputs.append(output)
        
        # in paper they also return the mask
        return outputs
    
class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, num_layers, padding_idx, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(num_layers, padding_idx, **kwargs)
        self.f = Sequential()
        self.f.add(Dense(self.output_size, activation='relu'))
        self.f.add(Dropout(self.dropout))
        self.f.add(LayerNormalization())
        
    @tf.function
    def call(self, input):
        output = self.f(input)
        output = super(MemoryAugmentedEncoder, self).call(output)
        return output