import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        
        self.fc = Sequential()
        self.fc.add(Dense(embedding_dim))
        self.fc.add(Dense(embedding_dim, activation='relu'))

    @tf.function
    def call(self, x):
        # Note that x represents the features of images, not the images themselves
        return self.fc(x)