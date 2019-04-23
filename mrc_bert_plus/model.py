# -*- coding: utf-8 -*-
"""
创建模型
"""
import tensorflow as tf
from bert import modeling, optimization
# from layers_plus.BiDAF_match_layer import *
# from layers_plus.BiDAF_pointer_net import *
# from layers_plus.basic_rnn import rnn
from layers_plus.rnet_pointer import *


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, query_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()
    # final_pooled = model.get_pooled_output()

    # final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    # batch_size = final_hidden_shape[0]
    # seq_length = final_hidden_shape[1]
    # hidden_size = final_hidden_shape[2]
    #
    # p_length, q_length = seq_length, seq_length

    # match_layer = AttentionFlowMatchLayer(hidden_size)
    # query_ids = tf.ones(tf.shape(segment_ids), tf.int32) - tf.cast(segment_ids, tf.int32)
    # match_p_encodes, _ = match_layer.match(final_hidden, final_hidden, segment_ids, query_ids, p_length, q_length)

    # fuse_p_encodes, _ = rnn('bi-lstm', match_p_encodes, [p_length]*batch_size,
    #                              hidden_size, layer_num=1)

    # pointerNetDecoder = PointerNetwork(hidden_size)
    # (start_logits, end_logits) = pointerNetDecoder.decode(match_p_encodes, fuse_p_encodes, segment_ids, p_length, batch_size)
    # is_train = tf.constant(False, dtype=tf.bool)
    # init = summ(final_hidden, hidden_size, query_ids, keep_prob=0.1, is_train=True)
    # pointer = ptr_net(batch=batch_size, hidden=hidden_size, keep_prob=0.1, is_train=True)
    # start_logits, end_logits = pointer(init, match_p_encodes, hidden_size, segment_ids)

    # output_weights = tf.get_variable(
    #     "cls/squad/output_weights", [2, hidden_size*2],
    #     initializer=tf.truncated_normal_initializer(stddev=0.02))
    #
    # output_bias = tf.get_variable(
    #     "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())
    #
    # final_hidden_matrix = tf.reshape(fuse_p_encodes,
    #                                  [batch_size * seq_length, hidden_size*2])
    # logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    # logits = tf.nn.bias_add(logits, output_bias)
    #
    # logits = tf.reshape(logits, [batch_size, seq_length, 2])
    # logits = tf.transpose(logits, [2, 0, 1])
    #
    # unstacked_logits = tf.unstack(logits, axis=0)
    #
    # (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    start_logits, end_logits = rnet(final_hidden, segment_ids, query_ids)
    return (start_logits, end_logits)


def rnet(final_hidden, segment_ids, query_ids):
    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    hidden_size = final_hidden_shape[2]

    c_len = tf.reduce_sum(tf.cast(segment_ids, tf.int32), axis=1)
    q_len = tf.reduce_sum(tf.cast(query_ids, tf.int32), axis=1)

    """
    训练时 
        keep_prob = 0.5
        is_train = tf.constant(True, dtype=tf.bool)
    测试时    
        keep_prob = 1.0
        is_train = tf.constant(False, dtype=tf.bool)
    
    """
    is_train = tf.constant(True, dtype=tf.bool)
    gru = native_gru

    keep_prob = 0.5
    hidden_size = 128

    with tf.variable_scope("attention"):
        # context-to-query attention, q_mak
        qc_att = dot_attention(final_hidden, final_hidden, mask=query_ids, hidden=hidden_size,
                               keep_prob=keep_prob, is_train=is_train)
        # qc_att gru
        rnn = gru(num_layers=1, num_units=hidden_size, batch_size=batch_size, input_size=qc_att.get_shape(
        ).as_list()[-1], keep_prob=keep_prob, is_train=is_train)
        # qc_att
        att = rnn(qc_att, seq_len=c_len)

    with tf.variable_scope("match"):
        self_att = dot_attention(
            att, att, mask=segment_ids, hidden=hidden_size, keep_prob=keep_prob, is_train=is_train)
        rnn = gru(num_layers=1, num_units=hidden_size, batch_size=batch_size, input_size=self_att.get_shape(
        ).as_list()[-1], keep_prob=keep_prob, is_train=is_train)
        match = rnn(self_att, seq_len=c_len)

    with tf.variable_scope("pointer"):
        init = summ(final_hidden[:, :, -2 * hidden_size:], hidden_size, mask=query_ids,
                    keep_prob=keep_prob, is_train=is_train)
        pointer = ptr_net(batch=batch_size, hidden=init.get_shape().as_list(
        )[-1], keep_prob=keep_prob, is_train=is_train)
        logits1, logits2 = pointer(init, match, hidden_size, segment_ids)

        return logits1, logits2
    # with tf.variable_scope("predict"):
    #     outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
    #                       tf.expand_dims(tf.nn.softmax(logits2), axis=1))
    #     outer = tf.matrix_band_part(outer, 0, 15)
    #     # yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
    #     # yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
    #     start_logits = tf.reduce_max(outer, axis=2)
    #     end_logits = tf.reduce_max(outer, axis=1)
    #
    #     return start_logits, end_logits
    # losses = tf.nn.softmax_cross_entropy_with_logits_v2(
    #     logits=logits1, labels=tf.stop_gradient(y1))
    # losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(
    #     logits=logits2, labels=tf.stop_gradient(y2))
    # loss = tf.reduce_mean(losses + losses2)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        query_ids = features["query_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            query_ids=query_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2.0

            ################################
            """
            loss Nan或波动剧烈解决思路：
                #1. 替换tf.nn.softmax_cross_entropy_with_logits_v2()为
                    tf.reduce_sum(y *tf.log(y_))即 compute_loss()中计算方法
                    需要注意logits的计算
                #2. 降低learning rate ==> 5e-5 5e-8
                #3. 降低batch_size ==> batch_size=1
                    后续在此基础上使用 batch_size = 8试试？？？
            Solutions:
                start_postions/end_positions 不能全为0
                删除没有答案的样本
            #################################
            Loss oscillating:
                1. 降低lr
                2. 增加batch size
                3. think:
                    增加文章最大长度
                
            """
            # start_positions = tf.one_hot(
            #     start_positions, depth=seq_length, dtype=tf.float32)
            # end_positions = tf.one_hot(
            #     end_positions, depth=seq_length, dtype=tf.float32)
            #
            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            #     logits=start_logits, labels=tf.stop_gradient(start_positions))
            # losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(
            #     logits=end_logits, labels=tf.stop_gradient(end_positions))
            #
            # total_loss = tf.reduce_mean(losses + losses2)
            ###############################

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=10)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:

            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn
