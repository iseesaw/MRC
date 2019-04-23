"""Example TensorFlow code for Self-Attention mechanism.

Refs:
    Attention Is All You Need
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    https://arxiv.org/abs/1706.03762

    Transformer: A Novel Neural Network Architecture for Language Understanding
    https://research.googleblog.com/2017/08/transformer-novel-neural-network.html

    tensor2tensor
    https://github.com/tensorflow/tensor2tensor
"""
import tensorflow as tf

_major_version, _minor_version, _ = map(int, tf.__version__.split('-')[0].split('.'))
assert _major_version >= 1 and _minor_version >= 2, "requires TensorFlow 1.2.0 and above"


def attention_fun(Q, K, scaled_=True, masked_=False):
    attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

    if scaled_:
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

    if masked_:
        raise NotImplementedError

    attention = tf.nn.softmax(attention, dim=-1)  # [batch_size, sequence_length, sequence_length]
    return attention


def input_fun(**config):
    data = tf.random_normal((
        config['batch_size'], config['sequence_length'], config['hidden_dim']))
    return data


def model_fun(data, **config):
    Q = tf.layers.dense(data, config['hidden_dim'])  # [batch_size, sequence_length, hidden_dim]
    K = tf.layers.dense(data, config['hidden_dim'])  # [batch_size, sequence_length, hidden_dim]
    V = tf.layers.dense(data, config['n_classes'])  # [batch_size, sequence_length, n_classes]

    attention = attention_fun(Q, K)  # [batch_size, sequence_length, sequence_length]
    output = tf.matmul(attention, V)  # [batch_size, sequence_length, n_classes]
    return output


if __name__ == '__main__':
    inputs = input_fun(batch_size=32, sequence_length=10, hidden_dim=128)
    outputs = model_fun(inputs, hidden_dim=128, n_classes=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        outputs_ = sess.run(outputs)
        print(outputs_.shape)
