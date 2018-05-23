"takes a string id and constructs a tensorflow graph, the fully-convolutional portion of a big CNN network. Generally from the TF Slim collection."

import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import inception

DEBUG = False


class BaseFCN:
	"Represents a pretrained fully convolutional network for plugging into other nets. Init gives a string identifier which pulls the net you want out of the tf-slim library.\
	After construction, these nets will not have their pre-trained weights. You need to call load_weights after running global initialization to initialize them with pretrained weights."
	def __init__(self,net_opts,inputs,is_train):
		#WARNING: don't give batch sizes of 1 if batch norm active
		
		if net_opts['base_net'] == 'resnet_v1':
			with slim.arg_scope(resnet_v1.resnet_arg_scope()) as scope:
				base_net, _ = resnet_v1.resnet_v1_152(inputs,is_training=False if net_opts['is_batchnorm_fixed'] else is_train,global_pool=False)
				if DEBUG:
					print('resnet_out:')
					print(resnet_out.shape.as_list())
				
				base_scope = 'resnet_v1_152'
		elif net_opts['base_net'] == 'resnet_v2':
			with slim.arg_scope(resnet_v2.resnet_arg_scope()) as scope:
				base_net, _ = resnet_v2.resnet_v2_152(inputs,is_training=False if net_opts['is_batchnorm_fixed'] else is_train,global_pool=False)
				if DEBUG:
					print('resnet_out:')
					print(resnet_out.shape.as_list())
				base_scope = 'resnet_v2_152'
		elif net_opts['base_net'] == 'inception_v3':
			#WARNING: is_train for inception controls not just batch norm but dropout. So it's a little awkward. We may need more functionality here later TODO
			
			#WARNING untested. Not sure I fully understand slim scopes yet
			base_scope = 'InceptionV3'
			with slim.arg_scope(inception_v3.inception_v3_arg_scope()) as scope:
				with variable_scope.variable_scope(scope, base_scope, [inputs, num_classes], reuse=reuse) as scope:
					with arg_scope([layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
						base_net, _ = inception_v3_base(inputs,scope=scope)
				
		elif net_opts['base_net'] == 'nothing':
			# a nothing for debugging purposes
			base_net = inputs
			base_scope = None
		else:
			raise Exception("basenet name not recognized")
			
		if base_scope:
			#for initializing with pre-trained values
			scope = ''.join([tf.contrib.framework.get_name_scope(), '/', base_scope])
			if DEBUG:
				print('scope:')
				print(scope)
			vars_to_restore = slim.get_variables_to_restore(include=[scope])
			if DEBUG:
				print('vars_to_restore:')
				print(vars_to_restore)
			self._pretrain_saver = tf.train.Saver(vars_to_restore)
		else:
			self._pretrain_saver = None
		
		self.out = base_net
		
	def load_weights(self,init_weight_fname,sess):
		"Initializes (or re-initializes) the pre-trained base model of the network from file. Should be called after running global initializer and before using."
		if self._pretrain_saver:
			self._pretrain_saver.restore(sess,init_weight_fname)