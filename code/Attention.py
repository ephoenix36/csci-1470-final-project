import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, units, kq_dim, v_dim, h, m):
        """
        Args:
            units (int): Output dimensionality
            kq_dim (int): Dimensionality of keys and queries
            v_dim (int): Dimensionality of values
            h (int): Number of heads
            m (int): Number of memory slots
        """
        super(MultiHeadAttention, self).__init__()
        
        self.units = units
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.h = h
        self.m = m
        
        self.wk = tf.keras.layers.Dense(h * kq_dim) # key
        self.wv = tf.keras.layers.Dense(h * kq_dim) # value
        self.wq = tf.keras.layers.Dense(h * v_dim) # query
        self.wo = tf.keras.layers.Dense(units)
        # in the paper they initialized these with a uniform distribution???
        
        # These are learnable params that get appended onto the end of wk and wv to help allowing us to learn knowledge about how two objects may be related
        self.mk = tf.Variable(tf.random.normal([1, m, h * kq_dim], stddev=.1)) 
        self.mv = self.mk = tf.Variable(tf.random.normal([1, m, h * v_dim], stddev=.1))
        # standard devisations in paper seem to be 1 / kq_dim and 1 / v_dim
    
    @tf.function
    def call(self, queries, keys, values, attention_mask=None):
        mk = tf.sqrt(self.mk) * self.mk.expand()
        # HEREEEEEEEEEEEEEE
        pass