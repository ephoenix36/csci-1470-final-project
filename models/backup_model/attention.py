import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.tanh = Activation('tanh')
        self.V = Dense(1)

    @tf.function
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        attention_hidden_layer = self.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))

        values = self.V(attention_hidden_layer)

        attention_weights = tf.nn.softmax(values, 1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, 1)

        return context_vector, attention_weights
