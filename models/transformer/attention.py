import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization


class MemoryMultiHeadAttention(Layer):

    def __init__(self, output_dim, kq_dim, v_dim, num_heads=10, mem_dim=100, dropout=0.1):
        """
        Args:
            output_dim (int): Output dimensionality
            kq_dim (int): Dimensionality of keys and queries
            v_dim (int): Dimensionality of values
            num_heads (int): Number of heads
            mem_dim (int): Number of memory slots
        """
        super(MemoryMultiHeadAttention, self).__init__()

        # Initialization of ScaledDotProductAttentionMemory
        self.output_dim = output_dim
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.mem_dim = mem_dim

        self.wk = Dense(num_heads * kq_dim)  # key
        self.wv = Dense(num_heads * kq_dim)  # value
        self.wq = Dense(num_heads * v_dim)  # query
        
        self.dense_1 = Dense(output_dim)
        self.dropout = Dropout(rate=dropout)
        self.layer_norm = LayerNormalization()
        
        # TODO: get rid of hard code

        # These are learnable params that get appended onto the end of wk and wv, allowing us to learn knowledge about how two features may be related
        self.mk = tf.Variable(tf.random.normal([1, mem_dim, 640])) # num_heads * kq_dim
        self.mv = tf.Variable(tf.random.normal([1, mem_dim, 640])) # num_heads * v_dim

    @tf.function
    def call(self, queries, keys, values, attention_mask=None):
        
        batch_size =  queries.shape[0]

        mk = tf.sqrt(float(self.kq_dim)) * tf.repeat(self.mk, batch_size, 0)
        mv = tf.sqrt(float(self.mem_dim)) * tf.repeat(self.mv, batch_size, 0)
        
        
        q = tf.transpose(tf.reshape(self.wq(queries), [-1, queries.shape[1], self.num_heads, self.kq_dim]), perm=[0, 2, 1, 3])  # batch_size, num_heads, nq, kq_dim
        k = tf.transpose(tf.reshape(tf.concat([self.wk(keys), mk], axis=1), [-1, ]), perm=[0, 2, 3, 1])  # batch_size, num_heads, kq_dim, nk
        v = tf.transpose(tf.concat([self.wv(values), mv], axis=1), perm=[0, 2, 1, 3])  # batch_size, num_heads nk, v_dim

        attention_matrix = tf.matmul(q, k) / tf.sqrt(self.kq_dim)  # batch_size, num_heads, nq, nk
        attention_matrix = tf.nn.softmax(attention_matrix)

        output = tf.transpose(tf.matmul(attention_matrix, v), perm=[0, 2, 1, 3])  # batch_size, nq, num_heads * v_dim
        output = self.wo(output)
        return output
