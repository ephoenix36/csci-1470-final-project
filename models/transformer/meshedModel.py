import numpy as np
import tensorflow as tf
from transformer.utils import *
from transformer.encoder import *
from transformer.decoder import *


class MeshedMemoryModel(tf.keras.Model):
    def __init__(self, vocab_size, max_sentence_len, num_layers, padding_index, output_size):
        super(MeshedMemoryModel, self).__init__()

        self.learning_rate = 0.01
        self.batch_size = 100
        self.hidden_size = 50
        self.output_size = output_size

        # TODO: fill in other args
        self.encoder = MemoryAugmentedEncoder(num_layers, padding_index)
        self.decoder = MeshedDecoder(vocab_size, max_sentence_len, num_layers, padding_index, output_size)

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM)

    @tf.function
    def call(self, images, seq):
        encoder_output, mask_encoder = self.encoder(images)
        decoder_output = self.decoder(seq, encoder_output, mask_encoder)
        return decoder_output
    
    def init_state(self):
        return [tf.zeros(self.batch_size, 0), None, None]

    def accuracy_function(self, prbs, labels, mask):
        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(
            tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def loss_function(self, prbs, labels, mask):
        return self.scce(labels, prbs, sample_weight=mask)
