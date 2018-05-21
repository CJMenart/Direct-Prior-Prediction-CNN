import tensorflow as tf

def activation_summary(x,name=None):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    if name is None:
        name = x.op.name
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.scalar(name + '/max', tf.reduce_max(x))
    tf.summary.scalar(name + '/min', tf.reduce_min(x))