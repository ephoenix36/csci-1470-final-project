import tensorflow as tf
from .attention import BahdanauAttention
from tensorflow.keras.layers import Embedding, GRU, Dense


class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.hidden_dim,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.ff1 = Dense(self.hidden_dim)
        self.ff2 = Dense(vocab_size)

        self.attention = BahdanauAttention(self.hidden_dim)

    @tf.function
    def call(self, input, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)

        input = self.embedding(input)

        input = tf.concat([tf.expand_dims(context_vector, 1), input], -1)

        output, state = self.gru(input)

        output = self.ff1(output)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.ff2(output)

        return output, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.hidden_dim))
