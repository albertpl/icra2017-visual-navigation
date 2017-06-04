import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def leaky_relu(x, alpha=0.2):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    return tf.maximum(x, alpha * x)


def siamese(x1, x2, out_dim, name):
    # flatten input
    x1_size = np.prod(x1.get_shape().as_list()[1:])
    x2_size = np.prod(x2.get_shape().as_list()[1:])
    assert x1_size == x2_size
    with tf.variable_scope(name):
        x1 = tf.reshape(x1, [-1, x1_size], name='x1')
        x2 = tf.reshape(x2, [-1, x1_size], name='x2')
        w = tf.get_variable('weight', (x1_size, out_dim), initializer=layers.variance_scaling_initializer())
        b = tf.get_variable('bias', (out_dim,), initializer=tf.constant_initializer())
        x1 = tf.matmul(x1, w) + b
        x1 = leaky_relu(x1)
        x2 = tf.matmul(x2, w) + b
        x2 = leaky_relu(x2)
        x = tf.concat(values=[x1, x2], axis=1)
    return x


def add_summary(writer, value_dict, global_step=0):
    if writer is None or len(value_dict) == 0:
        return
    value = [tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()]
    summary = tf.Summary(value=value)
    writer.add_summary(summary, global_step=global_step)
    return


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), "shape is not fully known"
    return out


def numel(x):
    return np.prod(x.get_shape().as_list())


def flat_grad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], axis=0)


def add_get_from_flat(var_list):
    return tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)


def add_assign_from_flat(var_list, var_v):
    size = [numel(v) for v in var_list]
    assert sum(size) == var_v.get_shape().as_list()[0]
    start = 0
    assign_ops = []
    for s, v in zip(size, var_list):
        shape = var_shape(v)
        assign_ops.append(tf.assign(v, tf.reshape(var_v[start:start + s], shape)))
        start += s
    return var_v, tf.group(*assign_ops)

