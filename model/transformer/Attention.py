import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, units, kq_dim, v_dim, h=8, m=100):
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
        
        # print("MHA units:", units)
        # print("MHA kq_dim:", kq_dim)
        # print("MHA v_dim", v_dim)

        self.wk = tf.keras.layers.Dense(h * kq_dim)  # key
        self.wv = tf.keras.layers.Dense(h * kq_dim)  # value
        self.wq = tf.keras.layers.Dense(h * v_dim)  # query
        self.wo = tf.keras.layers.Dense(units)
        # in the paper they initialized these with a uniform distribution???

        # These are learnable params that get appended onto the end of wk and wv to help allowing us to learn knowledge about how two objects may be related
        self.mk = tf.Variable(tf.random.normal([1, m, int(h * kq_dim)], stddev=.1))
        self.mv = self.mk = tf.Variable(tf.random.normal([1, m, int(h * v_dim)], stddev=.1))
        # standard devisations in paper seem to be 1 / kq_dim and 1 / v_dim
        print(self.mk.shape)
        print("odnawndaw", self.mv.shape)

    @tf.function
    def call(self, queries, keys, values, attention_mask=None):
        # TODO: see if attention mask is needed and add is so

        print(self.mk)
        mk = tf.sqrt(float(self.mk)) * tf.repeat(self.mk, [self.h * self.kq_dim], 0)
        mv = tf.sqrt(float(self.m)) * tf.repeat(self.mv, [self.h * self.kq_dim], 0)
        
        print("mk shape:", mk.shape)
        print("mv shape:", mv.shape)

        # TODO: check dims of following matrices
        q = tf.transpose(self.wq(queries), perm=[0, 2, 1, 3])  # batch, h, nq?, kq_dim
        print("q shape:", q.shape)
        k = tf.transpose(tf.concat([self.wk(keys), mk], axis=1), perm=[0, 2, 3, 1])  # batch, h, kq_dim, nk?
        v = tf.transpose(tf.concat([self.wv(values), mv], axis=1), perm=[0, 2, 1, 3])  # batch, h, nk?, v_dim

        attention_matrix = tf.matmul(q, k) / tf.sqrt(self.kq_dim)  # batch, h, nq?, nk?
        attention_matrix = tf.nn.softmax(attention_matrix)

        output = tf.transpose(tf.matmul(attention_matrix, v), perm=[0, 2, 1, 3])  # batch, nq, h*d_v
        output = self.wo(output)
        return output
