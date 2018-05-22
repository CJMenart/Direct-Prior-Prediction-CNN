"construct tf-optimizer from net_opts settings."

import tensorflow as tf


def optimizer_from_string(net_opts):
	if net_opts['optimizer_type'] == 'Adam':
			optimizer = tf.train.AdamOptimizer(net_opts['learn_rate'],epsilon=0.01)
	elif net_opts['optimizer_type'] == 'Momentum':
		optimizer = tf.train.MomentumOptimizer(net_opts['learn_rate'],
						net_opts['momentum'],
						use_nesterov=net_opts['is_using_nesterov'])
	elif net_opts['optimizer_type'] == 'Yellowfin':
		raise Exception('Yellowfin not implemented.')
		#optimizer = yellowfin.YFOptimizer(learning_rate=1e-3)
	else:
		raise Exception('Unrecognized optimizer type.')
		return -1
	return optimizer