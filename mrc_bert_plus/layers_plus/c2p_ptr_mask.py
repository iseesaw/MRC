# -*- coding: utf-8 -*-
"""
"""
import tensorflow as tf


class BiDAFAttention():
    """"""

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = tf.get_variable(
            "cls/squad/context_weights", [hidden_size, 1],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.q_weight = tf.get_variable(
            "cls/squad/query_weights", [hidden_size, 1],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        self.cq_weight = tf.get_variable(
            "cls/squad/context_query_weight", [1, 1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        self.bias = tf.get_variable(
            "cls/squad/bias",[1],
            initializer=tf.zeros_initializer()
        )

    def forward(self, c, q, c_mask, q_mask):
        """"""
        batch_size, c_len, _ = tf.shape(c)
        q_len = tf.shape(q)
        s = self.get_similarity_matrix(c, q)

        c_mask = tf.reshape(c_mask, [batch_size, c_len, 1])
        q_mask = tf.reshape(q_mask, [batch_size, 1, q_len])

        s1 = masked_softmax(s, q_mask, dim=2)
        s2 = masked_softmax(s, c_mask, dim=1)

        # batch_size c_len q_len x batch_size q_len hid_size => batch_size c_len hid_size
        a = tf.matmul(s1, q)
        b = tf.matmul(tf.matmul(s1, tf.transpose(s2, [1,2])), c)

        x = tf.concat([c, a, c*a, c*b], dim=2)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        """
        c_len, q_len = tf.shape(c)[1], tf.shape(q)[2]
        c = tf.nn.dropout(c, self.drop_prob) # batch_size c_len hidden
        q = tf.nn.dropout(q, self.drop_prob) # batch_size q_len hidden

        # batch_size c_len q_len
        s0 = tf.matmul(c, self.c_weight)
        s0 = tf.expand_dims([-1, -1, q_len])

        s1 = tf.matmul(q, self.q_weight)
        s1 = tf.expand_dims(tf.transpose(s1, [1, 2]), [-1, c_len, -1])

        s2 = tf.matmul(c*self.cq_weight, tf.transpose(q, [1, 2]))

        s = s0 + s1 + s2 + self.bias

        return s

def masked_softmax(inputs, mask, dim=-1, mask_value=-1e30):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0

    Result of taking masked softmax over the logits
    """
    mask = tf.cast(mask, tf.float32)

    masked_logits = inputs*mask + mask_value*(1 - mask)

    probs = tf.nn.softmax(masked_logits, dim)

    return probs