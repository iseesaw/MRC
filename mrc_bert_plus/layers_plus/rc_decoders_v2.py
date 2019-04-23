"""
decoders:
    PointerNetDecoder
    RecurrentMLPDecoder
    NoAnswerScoreDecoder

usage:
decoder = PointerNetDecoder(hidden_size=150) # hidden_size usually sets to be half of len(word_vec)
start_probs, end_probs = decoder.decode(pq_encodes, q_encodes)
"""

import tensorflow as tf
import tensorflow.contrib as tc
import sys


def custom_dynamic_rnn(cell, inputs, inputs_len, initial_state=None):
    """
    Implements a dynamic rnn that can store scores in the pointer network,
    the reason why we implements this is that the raw_rnn or dynamic_rnn function in Tensorflow
    seem to require the hidden unit and memory unit has the same dimension, and we cannot
    store the scores directly in the hidden unit.
    Args:
        cell: RNN cell
        inputs: the input sequence to rnn
        inputs_len: valid length
        initial_state: initial_state of the cell
    Returns:
        outputs and state
    """
    batch_size = tf.shape(inputs)[0]
    max_time = tf.shape(inputs)[1]

    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1, 0, 2]))
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    t0 = tf.constant(0, dtype=tf.int32)
    if initial_state is not None:
        s0 = initial_state
    else:
        s0 = cell.zero_state(batch_size, dtype=tf.float32)
    f0 = tf.zeros([batch_size], dtype=tf.bool)

    def loop_fn(t, prev_s, emit_ta, finished):
        """
        the loop function of rnn
        """
        cur_x = inputs_ta.read(t)
        scores, cur_state = cell(cur_x, prev_s)

        # copy through
        scores = tf.where(finished, tf.zeros_like(scores), scores)

        if isinstance(cell, tf.nn.rnn_cell.LSTMCell):
            cur_c, cur_h = cur_state
            prev_c, prev_h = prev_s
            cur_state = tc.rnn.LSTMStateTuple(tf.where(finished, prev_c, cur_c),
                                              tf.where(finished, prev_h, cur_h))
        else:
            cur_state = tf.where(finished, prev_s, cur_state)

        emit_ta = emit_ta.write(t, scores)
        finished = tf.greater_equal(t + 1, inputs_len)
        return [t + 1, cur_state, emit_ta, finished]

    _, state, emit_ta, _ = tf.while_loop(
        cond=lambda _1, _2, _3, finished: tf.logical_not(tf.reduce_all(finished)),
        body=loop_fn,
        loop_vars=(t0, s0, emit_ta, f0),
        parallel_iterations=32,
        swap_memory=False)

    outputs = tf.transpose(emit_ta.stack(), [1, 0, 2])
    return outputs, state


def attend_pooling(pooling_vectors, ref_vector, scope=None):
    """
    Applies attend pooling to a set of vectors according to a reference vector.
    Args:
        pooling_vectors: the vectors to pool shaped as [batch_size, p_length, word_vec_len]
        ref_vector: the reference vector shaped as [batch_size, word_vec_len]
        scope: score name
    Returns:
        the pooled vector
    """
    with tf.variable_scope(scope or 'attend_pooling'):
        p_len = pooling_vectors.get_shape().as_list()[-2]
        U = tf.tanh(pooling_vectors + tf.tile(tf.expand_dims(ref_vector, 1), [1, p_len, 1]))
        logits = tf.layers.dense(U, 1)
        scores = tf.nn.softmax(logits, 1)
        pooled_vector = tf.reduce_sum(pooling_vectors * scores, axis=1)
    return pooled_vector


class PointerNetLSTMCell(tf.nn.rnn_cell.LSTMCell):
    """
    Implements the Pointer Network Cell
    """
    def __init__(self, num_units, context_to_point):
        super().__init__(num_units, state_is_tuple=True)
        self.context_to_point = context_to_point
        # self.fc_context = tc.layers.fully_connected(self.context_to_point,
        #                                             num_outputs=self._num_units,
        #                                             activation_fn=None)
        self.fc_context = tf.layers.dense(self.context_to_point, self._num_units)

    def call(self, inputs, state, scope=None):
        (c_prev, m_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            '''
            U = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(m_prev,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None),
                                         1))
            logits = tc.layers.fully_connected(U, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_point * scores, axis=1)

            '''
            p_len = self.fc_context.get_shape().as_list()[-2]
            U = tf.tanh(self.fc_context + tf.tile(tf.expand_dims(m_prev, 1), [1, p_len, 1]))
            logits = tf.layers.dense(U, 1)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_point * scores, axis=1)
            lstm_out, lstm_state = super().__call__(attended_context, state)
        return tf.squeeze(scores, -1), lstm_state


class PointerNetDecoder(object):
    """
    Implements the Pointer Network
    """
    def __init__(self, h_size):
        self.hidden_size = h_size

    def decode(self, passage_vectors, question_vectors, init_with_question=True):
        """
        Use Pointer Network to compute the probabilities of each position
        to be start and end of the answer
        Args:
            passage_vectors: the encoded passage vectors
            question_vectors: the encoded question vectors
            init_with_question: if set to be true,
                             we will use the question_vectors to init the state of Pointer Network
        Returns:
            the probs of evary position to be start and end of the answer
        """
        with tf.variable_scope('pn_decoder'):
            fake_inputs = tf.zeros([tf.shape(passage_vectors)[0], 2, 1])  # not used
            sequence_len = tf.tile([2], [tf.shape(passage_vectors)[0]])
            if init_with_question:
                # random_attn_vector = tf.Variable(tf.random_normal([1, self.hidden_size]),
                #                                  trainable=True, name="random_attn_vector")
                random_attn_vector = tf.get_variable('random_attn_vector', [1, self.hidden_size], tf.float32,
                                                     tf.truncated_normal_initializer())
                # pooled_question_rep = tc.layers.fully_connected(
                #     attend_pooling(question_vectors, random_attn_vector),
                #     num_outputs=self.hidden_size, activation_fn=None
                # )
                pooled_question_rep = tf.layers.dense(attend_pooling(question_vectors, random_attn_vector),
                                                      self.hidden_size)
                init_state = tc.rnn.LSTMStateTuple(pooled_question_rep, pooled_question_rep)
                # init_state = tc.rnn.LSTMStateTuple(question_vectors, question_vectors)
            else:
                init_state = None
            with tf.variable_scope('fw'):
                fw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                fw_outputs, _ = custom_dynamic_rnn(fw_cell, fake_inputs, sequence_len, init_state)
            with tf.variable_scope('bw'):
                bw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                bw_outputs, _ = custom_dynamic_rnn(bw_cell, fake_inputs, sequence_len, init_state)
            start_prob = (fw_outputs[0:, 0, 0:] + bw_outputs[0:, 1, 0:]) / 2
            end_prob = (fw_outputs[0:, 1, 0:] + bw_outputs[0:, 0, 0:]) / 2
            return start_prob, end_prob


class RecurrentMLPDecoder(object):
    def __init__(self, h_s, m_s_l):
        self.hidden_size = h_s
        self.max_seq_len = m_s_l

    def decode(self, passage_vectors, init_vec):
        """
        Use simple a recurrent MLP to compute the probabilities of each position
        to be start and end of the answer.
        Args:
            passage_vectors: the encoded passage vectors.
            init_vec: usually set to be pooled question_vectors.
        Returns:
            the probs of every position to be start and end of the answer.
        """
        with tf.variable_scope('recurrentMLPDecoder'):
            # init_vec = tf.get_variable('init_vec', [self.max_seq_len], tf.float32,
            #                            tf.truncated_normal_initializer(), trainable=True)
            start_vec = attend_pooling(passage_vectors, init_vec, scope='start_position')
            end_vec = attend_pooling(passage_vectors, start_vec, scope='end_position')
            start_prob = tf.layers.dense(start_vec, self.max_seq_len, tf.nn.softmax)
            end_prob = tf.layers.dense(end_vec, self.max_seq_len, tf.nn.softmax)
            return start_prob, end_prob


class NoAnswerScoreDecoder(PointerNetDecoder):
    """
    Generate z-score as the no-answer probability along with the start & end probability.
    """
    def __init__(self, hidden_size):
        super().__init__(hidden_size)

    def decode(self, passage_vectors, question_vectors, init_with_question=True):
        with tf.variable_scope('decoder'):
            fake_inputs = tf.zeros([tf.shape(passage_vectors)[0], 2, 1])  # not used
            sequence_len = tf.tile([2], [tf.shape(passage_vectors)[0]])
            if init_with_question:
                random_attn_vector = tf.Variable(tf.random_normal([1, self.hidden_size]),
                                                 trainable=True, name="random_attn_vector")
                pooled_question_rep = tc.layers.fully_connected(
                    attend_pooling(question_vectors, random_attn_vector),
                    num_outputs=self.hidden_size, activation_fn=None
                )
                init_state = tc.rnn.LSTMStateTuple(pooled_question_rep, pooled_question_rep)
            else:
                init_state = None
            with tf.variable_scope('fw'):
                fw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                fw_outputs, fw_states = custom_dynamic_rnn(fw_cell, fake_inputs, sequence_len, init_state)
            with tf.variable_scope('bw'):
                bw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                bw_outputs, bw_states = custom_dynamic_rnn(bw_cell, fake_inputs, sequence_len, init_state)
            start_prob = (fw_outputs[0:, 0, 0:] + bw_outputs[0:, 1, 0:]) / 2
            end_prob = (fw_outputs[0:, 1, 0:] + bw_outputs[0:, 0, 0:]) / 2
            concat_states = tf.concat([fw_states[1], bw_states[1]], 1)
            hidden_states = tf.layers.dense(concat_states, self.hidden_size, tf.nn.relu)
            no_answer_prob = tf.layers.dense(hidden_states, 2, tf.nn.softmax)
            # s_position = tf.argmax(start_prob, 1)
            # e_position = tf.argmax(end_prob, 1)
        return start_prob, end_prob, no_answer_prob


if __name__ == '__main__':
    batch_size = 8
    max_p_l = 5
    word_vec_len = 3
    hidden_size = 3
    pq_encodes = tf.get_variable('pq_encodes', [batch_size, max_p_l, word_vec_len], tf.float32,
                                 tf.random_normal_initializer)
    q_encodes = tf.get_variable('p_encodes', [batch_size, word_vec_len], tf.float32,
                                tf.random_normal_initializer)
    decoder = PointerNetDecoder(hidden_size)
    # decoder = RecurrentMLPDecoder(hidden_size, max_p_l)
    start_probs, end_probs = decoder.decode(pq_encodes, q_encodes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(start_probs))
        print('start_probs.get_shape().as_list() = ', start_probs.get_shape().as_list())
