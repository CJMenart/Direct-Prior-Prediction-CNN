"Encapsulates adding a number of useful Tensorboard summaries to a matrix tensor."
import tensorflow as tf

def activation_summary(x,name=None):
    "\
    :param x: A Tensor\
	:param name: optional name that will show up in Tensorboard. Recommended if you have many summaries.\
    :return: Add histogram summary and scalar summary of the sparsity of the tensor\
    "
    if name is None:
        name = x.op.name
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.scalar(name + '/max', tf.reduce_max(x))
    tf.summary.scalar(name + '/min', tf.reduce_min(x))