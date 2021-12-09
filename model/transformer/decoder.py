import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout

from transformer.attention import MemoryMultiHeadAttention
from transformer.utils import *

class MeshedDecoderLayer(Layer):
    def __init__(self, vocab_size, max_sentence_len, padding_index, output_size, kq_size=64, v_size=64, hidden_dim=1024, dropout=0.1):
        super(MeshedDecoderLayer, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_sentence_len = max_sentence_len
        self.padding_index = padding_index
        self.output_size = output_size
        self.kq_size = kq_size
        self.v_size = v_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.self_attention = MemoryMultiHeadAttention(output_size, kq_size, v_size, dropout=dropout)
        self.encoder_attention = MemoryMultiHeadAttention(output_size, kq_size, v_size, dropout=dropout)
        
        self.pwff = PositionWiseFeedForward(output_size, hidden_dim, dropout)
        
        self.alpha1 = Dense(output_size, activation='sigmoid')
        self.alpha2 = Dense(output_size, activation='sigmoid')
        self.alpha3 = Dense(output_size, activation='sigmoid')
    
    @tf.function
    def call(self, input, encoder_output, mask_pad, mask_self_attention, mask_encoder_attention):
        self_att = self.self_attention(input, input, input, mask_self_attention)
        self_att *= mask_pad
        
        encoder_att1 = self.encoder_attention(self_att, encoder_output[:, 0], encoder_output[:, 0], mask_encoder_attention)
        encoder_att2 = self.encoder_attention(self_att, encoder_output[:, 1], encoder_output[:, 1], mask_encoder_attention)
        encoder_att3 = self.encoder_attention(self_att, encoder_output[:, 2], encoder_output[:, 2], mask_encoder_attention)
        
        alpha1 = self.alpha1(tf.concat([self_att, encoder_att1], -1))
        alpha2 = self.alpha2(tf.concat([self_att, encoder_att2], -1))
        alpha3 = self.alpha3(tf.concat([self_att, encoder_att3], -1))
        
        encoder_attention = (encoder_att1 * alpha1 + encoder_att2 * alpha2 + encoder_att3 * alpha3) / tf.sqrt(3)
        encoder_attention * mask_pad
        
        output = self.pwff(encoder_attention)
        output = self.pwff * mask_pad
        return output
    
class MeshedDecoder(Layer):
    def __init__(self, vocab_size, max_sentence_len, num_layers, padding_index, output_size, word_embedding_size=128, kq_size=64, v_size=64, hidden_size=1024, dropout=0.1):
        super(MeshedDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_sentence_len = max_sentence_len
        self.num_layers = num_layers
        self.padding_index = padding_index
        self.output_size = output_size
        self.kq_size = kq_size
        self.v_size = v_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # TODO: POTENTIAL SOURCE OF ERROR
        self.word_embeddings = tf.Variable(tf.random.normal(
            [vocab_size, int(word_embedding_size)], stddev=.1, dtype=tf.float32))
        self.positional_embeddings = Position_Encoding_Layer(max_sentence_len + 1, output_size)
        
        self.layers = [MeshedDecoderLayer(vocab_size, max_sentence_len, padding_index, output_size, kq_size, v_size, hidden_size, dropout) for _ in range(num_layers)]
        
        self.f = Dense(vocab_size, use_bias=False, activation='softmax')
    

    @tf.function
    def call(self, input, encoder_output, mask_encoder_attention):
        batch_size, seq_len = input.shape[:2]
        
        # TODO: check/fix masks
        mask_queries = tf.expand_dims(input != self.padding_index, -1)
        
        mask_self_attention = tf.experimental.numpy.triu(tf.ones(seq_len, seq_len))
        mask_self_attention = tf.expand_dims(tf.expand_dims(mask_self_attention, 0))
        mask_self_attention = tf.expand_dims(tf.expand_dims(input == self.padding_index, 1), 1)
        
        seq = tf.repeat(tf.reshape(tf.range(1, seq_len + 1), [1, -1]), batch_size, 0)
        seq *= tf.boolean_mask(input, tf.squeeze(mask_queries, -1) == 0)
        
        output = self.word_embeddings(input) + self.positional_embeddings(seq)
        for layer in self.layers:
            output = layer(output, encoder_output, mask_queries, mask_self_attention, mask_encoder_attention)
        
        output = self.f(output)
        return output