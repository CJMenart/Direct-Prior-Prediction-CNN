"Wraps around a base network (ResNet or Inception) and predicts a distribution directly from passed-in image"

from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
#also import inception TODO
import tensorflow as tf
from activation_summary import *
import os
import numpy as np
from loss_functions import *
import prob_remapping

slim = tf.contrib.slim #slim is used by the official TF implementation of ResNet
DAT_TYPE = tf.float32
DEBUG = True

#TODO: Make nets inherit from a 'net' class with certain public things exposed


#WARNING: Must not be called while inside another scope. Until I fix the include just a min bro TODO
class PriorNet:

	def __init__(self,net_opts):
		self.inputs = tf.placeholder(DAT_TYPE,[None,None,None,3])
		self.is_train = tf.placeholder(tf.bool,None)
		# we are loading segmentation maps in case we change this later to be more advanced. Future-proofing. SIGH TODO include option for simple priors to shave off load time?
		# this should be sufficient for computing prior targets. All 'black box' targets will be computed using actual remapping, so 
		# any other component I can think of would come directly from this label map
		# as this is only used for binary priors (even then not always?) consider making it internal, some kind of rolling avg
		self.seg_target = tf.placeholder(tf.int64,[None,None,None]) #batch,width,height
		self.processed_seg_target = self._expand_target(self.seg_target,net_opts)
		self.prior_target = self._pool_prior_target(self.processed_seg_target,net_opts)
		#below three are not necessary to feed if you don't use remapping loss or run seg_acc
		self.remap_target = tf.placeholder(tf.int64,[None,None])
		self.remap_base_prob = tf.placeholder(DAT_TYPE,[None,None,net_opts['num_labels']])		
		self.map_mat = tf.placeholder(DAT_TYPE,[net_opts['num_labels'],net_opts['num_labels']])
		'''
		this can just be one if you don't want weighted loss. This idea is that the average loss on any class should be 1, so we assign
		a greater loss than one to false-positive if it occurs more than half the time, etc. The value passed in should be the 1/2 the loss for false
		positives, and the false-negative loss is just 2-that.
		Note that code regularizes b/c according to this scheme if something showed up in every image the penalty for false negative is 0...just sayin'
		'''
		self.class_frequency = tf.placeholder(DAT_TYPE,[1,net_opts['num_labels']])
		
		#WARNING: 'is_training' option now actually used--don't give batch sizes of 1. Batch norms insides
		#TODO: consider changing the type of pooling? Max pooling or something? I'd concat both but too many params
		if net_opts['base_net'] == 'resnet_v1':
			with slim.arg_scope(resnet_v1.resnet_arg_scope()) as scope:
				resnet_out, _ = resnet_v1.resnet_v1_152(self.inputs,is_training=False if net_opts['is_batchnorm_fixed'] else self.is_train,global_pool=True)
				if DEBUG:
					print('resnet_out:')
					print(resnet_out.shape.as_list())
				self.base_net = tf.squeeze(resnet_out,[1,2])
				base_scope = 'resnet_v1_152'
		elif net_opts['base_net'] == 'resnet_v2':
			with slim.arg_scope(resnet_v2.resnet_arg_scope()) as scope:
				resnet_out, _ = resnet_v2.resnet_v2_152(self.inputs,is_training=False if net_opts['is_batchnorm_fixed'] else self.is_train,global_pool=True)
				if DEBUG:
					print('resnet_out:')
					print(resnet_out.shape.as_list())
				self.base_net = tf.squeeze(resnet_out,[1,2])
				base_scope = 'resnet_v2_152'
		elif net_opts['base_net'] == 'inception':
			with slim.arg_scope(inception_arg_scope()):
				logits, end_points = inception.inception_v4(self.inputs, num_classes=None, is_training=False if net_opts['is_batchnorm_fixed'] else self.is_train)
				self.base_net = logits
				base_scope = 'InceptionV4'
		else:
			print('Error: basenet not recognized.')
			quit()
		
		activation_summary(self.base_net)

		#for initializing with pre-trained values
		#with some abuse of terminology
		scope = os.path.join(tf.contrib.framework.get_name_scope(), base_scope)
		#print('scope:')
		#print(scope)
		vars_to_restore = slim.get_variables_to_restore(include=[scope])
		#print('vars_to_restore:')
		#print(vars_to_restore)
		self._pretrain_saver = tf.train.Saver(vars_to_restore)

		self.prior = self._prior_predictor(net_opts)
		self.direct_prior_loss = self._weighted_prior_loss(net_opts)
		
		#What's interesting is that if we didn't have to save and load softmax, but were performing remapping using a model in tensorflow that we ran, this memory limit
		#wouldn't really exist. When we're loading off disk, we have to use 100-pixel examples, but we could remap the whole thing with an integrated model, it's much faster.
		#but right now we're refining a matconvnet model...still, keep this in mind for future
		if net_opts['remapping_loss_weight'] > 0:
			self.remapping_loss,self.seg_acc,_ = prob_remapping.conf_remap_loss_and_metrics(self.map_mat,self.prior,self.remap_base_prob,self.remap_target,net_opts['batch_size'],net_opts['epsilon'])
		else:
			self.seg_acc = tf.constant(-1.0,shape=None,dtype=DAT_TYPE)
		if net_opts['remapping_loss_weight'] > 0:
			self.loss = self.direct_prior_loss + net_opts['remapping_loss_weight']*self.remapping_loss
		else:
			self.loss = self.direct_prior_loss
		
		if net_opts['is_target_distribution']:
			self.prior_err = thresh_err(self.prior,self.prior_target,net_opts['err_thresh'])
		else:
			self.prior_err = hamming_err(self.prior,self.prior_target)
	
	def _prior_predictor(self,net_opts):
		#prior predictor here is very simple. Assume network ended in a global average pooling operation and add 1 FC layer, non-linearity

		in_feat = self.base_net
		in_chann = in_feat.shape.as_list()[-1]
		out_chann = net_opts['hid_layer_width']
		for lay in range(net_opts['num_hid_layers']):
			with tf.variable_scope('fc_%d' % lay) as scope:
				weights = tf.get_variable('weights',[in_chann,out_chann],DAT_TYPE,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight']))
				biases = tf.Variable(tf.constant(0.01, shape=[out_chann], dtype=DAT_TYPE),
									 trainable=True, name='biases')
				
				#TODO: Add batch-norm here. Be a bit of a pain in TF b/c you'd have to add the update ops for those layers to train, but not the other resnet ones...
				pre_act = tf.matmul(in_feat,weights)
				pre_act = tf.nn.bias_add(pre_act, biases)
				activation = tf.nn.relu(pre_act, name='activation')
				activation = tf.nn.dropout(activation,net_opts['dropout_prob'])
				in_feat = activation
				in_chann = in_feat.shape.as_list()[-1]
				
				tf.add_to_collection('fresh',weights)
				tf.add_to_collection('fresh',biases)
				
		out_chann = net_opts['num_labels']	
		with tf.variable_scope('fc_final') as scope:
			weights = tf.get_variable('weights',[in_chann,out_chann],DAT_TYPE,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight']))
			biases = tf.Variable(tf.constant(0.01, shape=[out_chann], dtype=DAT_TYPE),
								 trainable=True, name='biases')
			
			pre_act = tf.matmul(in_feat,weights)
			pre_act = tf.nn.bias_add(pre_act, biases)
			
			if net_opts['is_target_distribution']:
				activation = tf.nn.relu(pre_act, name='activation')
				activation = activation/tf.reduce_sum(activation,-1)
			else: #assumed that we basically have binary target if not distribution
				activation = tf.nn.sigmoid(pre_act)
			
			tf.add_to_collection('fresh',weights)
			tf.add_to_collection('fresh',biases)
		
			return activation
		
	def load_weights(self,init_weight_fname,sess,net_opts):
		"Initializes (or re-initializes) the pre-trained base model of the network from file. Should be called after running global initializer and before using."
		self._pretrain_saver.restore(sess,init_weight_fname)

	def _weighted_prior_loss(self,net_opts):
		#specific loss we want to use in the case of binary prior. 
		if net_opts['is_target_distribution']:
			return -1 #TODO ask Mo for good loss. Probably select one of several
		else:
			#Cross-entropy loss. TODO factor out 
			false_pos_weight = self.class_frequency*2+net_opts['epsilon']
			false_neg_weight = 2-self.class_frequency*2+net_opts['epsilon']
			if DEBUG:
				print('false_neg_weights,self.prior_target,self.prior:')
				print(false_neg_weight.shape.as_list())
				print(self.prior_target.shape.as_list())
				print(self.prior.shape.as_list())
			
			loss = tf.reduce_mean(-false_neg_weight*self.prior_target*tf.log(tf.maximum(self.prior,net_opts['epsilon']))) + tf.reduce_mean(-false_pos_weight*(1-self.prior_target)*tf.log(tf.maximum(1-self.prior,net_opts['epsilon'])))
			return loss
			
	#downsamples the processed 4d target tensor to form a prior. Can form binary or histogram prior
	def _pool_prior_target(self,target,net_opts):
		if net_opts['is_target_distribution']:
			target = tf.reduce_max(target,[1,2])
		else:
			target = tf.reduce_mean(target,[1,2])
		return target
		
	#expands a 2d map of the correct classes for a tensor and expands to 3d one-hot vectors
	#tensors go from 3d to 4d--batch dim at beginning
	def _expand_target(self,target,net_opts): 
		target = tf.one_hot(target,net_opts['num_labels']+1,dtype=DAT_TYPE,axis=3)
		target = target[:,:,:,1:]
		return target
		
	