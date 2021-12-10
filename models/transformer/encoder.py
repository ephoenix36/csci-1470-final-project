import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout

from transformer.attention import MemoryMultiHeadAttention
from transformer.utils import *


class EncoderLayer(Layer):
    def __init__(self, output_size, key_size=512, value_size=512, hidden_size=1024, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.attention = MemoryMultiHeadAttention(output_size, key_size, value_size, dropout=dropout)

        self.pwff = PositionWiseFeedForward(output_size, hidden_size, dropout)

    @tf.function
    def call(self, queries, keys, values, attention_mask):
        output = self.attention(queries, keys, values, attention_mask)
        output = self.pwff(output)
        return output


class MultiLevelEncoder(Layer):
    def __init__(self, num_layers, padding_index, output_size=512, hidden_size=1024, dropout=0.1):
        super(MultiLevelEncoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.layers = [EncoderLayer(output_size, hidden_size, dropout) for _ in range(num_layers)]

        self.padding_index = padding_index

    @tf.function
    def call(self, input):
        attention_mask = tf.reduce_sum(input, -1) == self.padding_index
        attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 1)

        outputs = []
        output = input

        for layer in self.layers:
            output = layer(output, output, output, attention_mask)
            outputs.append(output)

        return outputs, attention_mask


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
        return super(MemoryAugmentedEncoder, self).call(output)
