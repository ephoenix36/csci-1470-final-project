import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from .attention import *
from .utils import *

class MeshedDecoderLayer(Layer):
    def __init__(self, vocab_size, max_sentence_len, padding_index, output_size, kq_size=64, v_size=64, hidden_size=1024, dropout=0.1):
        super(MeshedDecoderLayer, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_sentence_len = max_sentence_len
        self.padding_index = padding_index
        self.output_size = output_size
        self.kq_size = kq_size
        self.v_size = v_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # TODO: look into can_be_stateful field
        self.self_attention = MultiHeadAttention(output_size, kq_size, v_size)
        self.encoder_attention = MultiHeadAttention(output_size, kq_size, v_size)
        
        self.ff = PositionWiseFeedForward(output_size, hidden_size, dropout)
        
        self.alpha1 = Dense(output_size, activation='sigmoid')
        self.alpha2 = Dense(output_size, activation='sigmoid')
        self.alpha3 = Dense(output_size, activation='sigmoid')
    
    # TODO: try and understand what is going on here
    @tf.function
    def call(self, input, encoder_output, mask_pad, mask_self_attention, mask_encoder_attention):
        self_att = self.self_attention(input, input, input)
        self_att *= mask_pad
        
        encoder_att1 = self.encoder_attention(self_att, encoder_output[:, 0], encoder_output[:, 0])
        encoder_att2 = self.encoder_attention(self_att, encoder_output[:, 1], encoder_output[:, 1])
        encoder_att3 = self.encoder_attention(self_att, encoder_output[:, 2], encoder_output[:, 2])
        
        alpha1 = self.alpha1(tf.concat([self_att, encoder_att1], -1))
        alpha2 = self.alpha2(tf.concat([self_att, encoder_att2], -1))
        alpha3 = self.alpha3(tf.concat([self_att, encoder_att3], -1))
        
        encoder_attention = (encoder_att1 * alpha1 + encoder_att2 * alpha2 + encoder_att3 * alpha3) / tf.sqrt(3)
        encoder_attention * mask_pad
        
        output = self.ff(encoder_attention)
        output = self.ff * mask_pad
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
        
        # TODO: add word and positional embeddings
        self.word_embeddings = tf.Variable(tf.random.normal(
            [vocab_size, word_embedding_size], stddev=.1, dtype=tf.float32))
        self.positional_embeddings = Position_Encoding_Layer(max_sentence_len + 1, output_size)
        
        self.layers = [MeshedDecoder(vocab_size, max_sentence_len, padding_index, output_size, kq_size, v_size, hidden_size, dropout) for _ in range(num_layers)]
        
        self.f = Dense(vocab_size, use_bias=False, activation='softmax')
    

    @tf.function
    def call(self, input, encoder_output, mask_encoder_attention):
        seq_len = input.shape[:1]
        
        # TODO: check/fix masks
        mask_queries = input == self.padding_index
        
        mask_self_attention = tf.experimental.numpy.tril(tf.ones(seq_len, seq_len))
        mask_self_attention += input == self.padding_index
        
        seq = tf.boolean_mask(input, mask_queries)
        
        output = self.word_embeddings(input) + self.positional_embeddings(seq)
        for layer in self.layers:
            output = layer(output, encoder_output, mask_queries, mask_self_attention, mask_encoder_attention)
        
        output = self.f(output)
        return output