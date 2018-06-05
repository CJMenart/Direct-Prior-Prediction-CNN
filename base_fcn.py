"takes a string id and constructs a tensorflow graph, the fully-convolutional portion of a big CNN network. Generally from the TF Slim collection."

import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import inception
from imagenet_names import imagenet_names
import cv2
import numpy as np

_VGG_R_MEAN = 123.68
_VGG_G_MEAN = 116.78
_VGG_B_MEAN = 103.94

DEBUG = False


class BaseFCN:
	"Represents a pretrained fully convolutional network for plugging into other nets. Init gives a string identifier which pulls the net you want out of the tf-slim library.\
	After construction, these nets will not have their pre-trained weights. You need to call load_weights after running global initialization to initialize them with pretrained weights."
	def __init__(self,net_opts,inputs,is_train):
		#WARNING: don't give batch sizes of 1 if batch norm active
		
		if net_opts['base_net'] == 'resnet_v1':
			with slim.arg_scope(resnet_v1.resnet_arg_scope()) as scope:
				inputs = _vgg_preprocess(inputs)
				base_net, _ = resnet_v1.resnet_v1_152(inputs,is_training=False if net_opts['is_batchnorm_fixed'] else is_train,global_pool=False)
				if DEBUG:
					print('resnet_out:')
					print(base_net.shape.as_list())
				
				base_scope = 'resnet_v1_152'
		elif net_opts['base_net'] == 'resnet_v2':
			with slim.arg_scope(resnet_v2.resnet_arg_scope()) as scope:
				inputs = _vgg_preprocess(inputs)
				base_net, _ = resnet_v2.resnet_v2_152(inputs,is_training=False if net_opts['is_batchnorm_fixed'] else is_train,global_pool=False)
				if DEBUG:
					print('resnet_out:')
					print(base_net.shape.as_list())
				base_scope = 'resnet_v2_152'
		elif net_opts['base_net'] == 'inception_v3':
			#WARNING: is_train for inception controls not just batch norm but dropout. So it's a little awkward. We may need more functionality here later TODO

			'''
			TODO add inception preprocessing: https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/preprocessing_factory.py
			
			#WARNING untested. Not sure I fully understand slim scopes yet
			base_scope = 'InceptionV3'
			with slim.arg_scope(inception.inception_v3_arg_scope()) as scope:
				with slim.variable_scope(scope, base_scope, [inputs, None], reuse=False) as scope:
					with slim.arg_scope([layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
						base_net, _ = inception_v3_base(inputs,scope=scope)
			'''

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


def _vgg_preprocess(inputs):
	return inputs - [[[[_VGG_R_MEAN,_VGG_G_MEAN,_VGG_B_MEAN]]]]


def test_base_fcn(net_name,img_fnames,weight_fname):
	"Quick effort to test the base fcns and figure out what preprocessing is appropriate--isn't well documented."
	inputs = tf.placeholder(tf.float32,[None,None,None,3])
	inputs = _vgg_preprocess(inputs)

	if net_name == 'resnet_v1_152':
		with slim.arg_scope(resnet_v1.resnet_arg_scope()) as scope:
			base_net, _ = resnet_v1.resnet_v1_152(inputs,
												  is_training=False,
												  num_classes=1000)
	else:
		raise Exception('net_name not recognized.')

	pred = tf.argmax(base_net,-1)
	saver = tf.train.Saver()
	sess = tf.InteractiveSession()
	#tf.global_variables_initializer().run()
	saver.restore(sess,weight_fname)
	for img_fname in img_fnames:
		img = cv2.imread(img_fname)
		prediction = sess.run(pred,feed_dict={inputs:img[np.newaxis,:,:,:]})
		print('pred for %s:' % img_fname)
		print(imagenet_names[int(prediction)])
		print('=========')