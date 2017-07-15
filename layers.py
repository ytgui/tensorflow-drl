import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def fc(x, n_neurons, activation=None):
    rows, cols = x.shape
    W = weight_variable([int(cols), n_neurons])
    b = bias_variable([n_neurons])
    if activation is None:
        y = tf.matmul(x, W) + b
    else:
        y = activation(tf.matmul(x, W) + b)
    return y, W, b