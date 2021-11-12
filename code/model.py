import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer_utils import mark_as_return


class MeshedMemoryModel(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
        ###### DO NOT CHANGE ##############
        super(RNN_Seq2Seq, self).__init__()
        self.french_vocab_size = french_vocab_size  # The size of the french vocab
        self.english_vocab_size = english_vocab_size  # The size of the english vocab

        self.french_window_size = french_window_size  # The french window size
        self.english_window_size = english_window_size  # The english window size
        ######^^^ DO NOT CHANGE ^^^##################

        # TODO:
        # 1) Define any hyperparameters
        learning_rate = 0.01

        # Define batch size and optimizer/learning rate
        self.batch_size = 100
        embedding_size = 100
        hidden_size = 50

        # 2) Define embeddings, encoder, decoder, and feed forward layers
        self.french_embeddings = tf.Variable(tf.random.normal(
            [self.french_vocab_size, embedding_size], stddev=.1, dtype=tf.float32))
        self.encoder_lstm = tf.keras.layers.LSTM(
            hidden_size, return_sequences=True, return_state=True)

        self.english_embeddings = tf.Variable(tf.random.normal(
            [self.english_vocab_size, embedding_size], stddev=.1, dtype=tf.float32))
        self.decoder_lstm = tf.keras.layers.LSTM(
            hidden_size, return_sequences=True, return_state=True)
        self.dense1 = tf.keras.layers.Dense(200, activation='relu')
        # self.dense2 = tf.keras.layers.Dense(200, activation='relu')
        self.dense2 = tf.keras.layers.Dense(
            self.english_vocab_size, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM)

    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        # TODO:
        # 1) Pass your french sentence embeddings to your encoder
        encoder_input = tf.nn.embedding_lookup(
            self.french_embeddings, encoder_input)

        # batch_size x french_window_size x french_vocab_size
        seq_output, memory, carry = self.encoder_lstm(encoder_input)
        # print(seq_output.shape)
        # print(memory.shape)
        # print(carry.shape)

        # 2) Pass your english sentence embeddings, and final state of your encoder, to your decoder

        # attention = tf.zeros((seq_output.shape[0], self.french_window_size, self.french_window_size))
        # for batch in range(seq_output.shape[0]):
        # 	for word1 in range(self.french_window_size):
        # 		for word2 in range(self.french_window_size):
        # 			attention[batch, word1, word2] = tf.matmul(tf.transpose(seq_output[batch, word1]), seq_output[batch, word2])
        # attention = tf.nn.softmax(attention, axis=1) # TODO: check axis

        # attention_seq_output = tf.zeros(seq_output.shape)
        # for batch in range(seq_output.shape[0]):
        # 	for word1 in range(self.french_window_size):
        # 		for weight in attention[batch, word1]:
        # 			for word2 in seq_output[batch]:
        # 				attention_seq_output[batch, word1] += weight * word2

        decoder_input = tf.nn.embedding_lookup(
            self.english_embeddings, decoder_input)

        seq_output, memory, carry = self.decoder_lstm(decoder_input, initial_state=(
            memory, carry))  # batch_size x english_window_size x english_vocab_size
        # initial_state=(memory, carry)

        # 3) Apply dense layer(s) to the decoder out to generate probabilities
        logits = self.dense1(seq_output)
        logits = self.dense2(logits)
        # logits = self.dense3(logits)

        return logits

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

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
        # prbs = tf.boolean_mask(prbs, mask)
        # labels = tf.boolean_mask(labels, mask)
        # return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, sample_weight=mask))

        return self.scce(labels, prbs, sample_weight=mask)
