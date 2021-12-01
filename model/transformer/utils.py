import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout

class PositionWiseFeedForward(Layer):
    def __init__(self, output_size, hidden_size=1024, dropout=0.1):
        self.ff = Sequential()
        self.ff.add(Dense(hidden_size, activation='relu'))
        self.ff.add(Dropout(dropout))
        self.ff.add(Dense(output_size))
        self.ff.add(Dropout(dropout))
        self.normalization = LayerNormalization()
        
    @tf.function
    def call(self, input):
        output = self.ff(input)
        output = self.normalization(output + input) # positional encoding
        return output
    
class Position_Encoding_Layer(tf.keras.layers.Layer):
    def __init__(self, window_sz, emb_sz):
        super(Position_Encoding_Layer, self).__init__()
        self.positional_embeddings = self.add_weight(
            "pos_embed", shape=[window_sz, emb_sz])

    @tf.function
    def call(self, x):
        """
        Adds positional embeddings to word embeddings.    

        :param x: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
        :return: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
        """
        return x+self.positional_embeddings