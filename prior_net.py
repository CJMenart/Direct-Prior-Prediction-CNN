"Wraps around a base network (ResNet or Inception) and predicts a distribution directly from passed-in image"

import tensorflow as tf
from activation_summary import *
import os
import numpy as np
from loss_functions import *
import prob_remapping
from base_fcn import *
from nn_util import *

slim = tf.contrib.slim #slim is used by the official TF implementation of ResNet
DAT_TYPE = tf.float32
DEBUG = True

#TODO: Make nets inherit from a 'net' class with certain public things exposed


#WARNING: Must not be called while inside another scope. Until I fix the include just a min bro TODO
class PriorNet:

	def __init__(self,net_opts,num_labels):
		self.inputs = tf.placeholder(DAT_TYPE,[None,None,None,3])
		self.is_train = tf.placeholder(tf.bool,None)
		# we are loading segmentation maps in case we change this later to be more advanced. Future-proofing. SIGH TODO include option for simple priors to shave off load time?
		# this should be sufficient for computing prior targets. All 'black box' targets will be computed using actual remapping, so 
		# any other component I can think of would come directly from this label map
		# as this is only used for binary priors (even then not always?) consider making it internal, some kind of rolling avg
		self.seg_target = tf.placeholder(tf.int64,[None,None,None]) #batch,width,height
		self.processed_seg_target = self._expand_target(self.seg_target,num_labels)
		self.prior_target = self._pool_prior_target(self.processed_seg_target,net_opts)
		#below three are not necessary to feed if you don't use remapping loss or run seg_acc
		self.remap_target = tf.placeholder(tf.int64,[None,None])
		self.remap_base_prob = tf.placeholder(DAT_TYPE,[None,None,num_labels])		
		self.map_mat = tf.placeholder(DAT_TYPE,[num_labels,num_labels])
		'''
		this can just be one if you don't want weighted loss. This idea is that the average loss on any class should be 1, so we assign
		a greater loss than one to false-positive if it occurs more than half the time, etc. The value passed in should be the 1/2 the loss for false
		positives, and the false-negative loss is just 2-that.
		Note that code regularizes b/c according to this scheme if something showed up in every image the penalty for false negative is 0...just sayin'
		'''
		self.class_frequency = tf.placeholder(DAT_TYPE,[1,num_labels])
		
		self._base_net = BaseFCN(net_opts,self.inputs,self.is_train)
		#TODO: consider changing the type of pooling? Max pooling or something? I'd concat both but too many params
		self._base_net_vectorized = tf.reduce_max(self._base_net.out,[1,2])
		
		activation_summary(self._base_net_vectorized,'base_net_vectorized')


		self.prior = self._prior_predictor(net_opts,num_labels)
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
	
	def _prior_predictor(self,net_opts,num_labels):
		"portion of the network that we add. Very simpe. Assume base_fcn is pooled into something vector-like, add FC layers, non-linearity, other whistles, then cap to number of classes."

		in_feat = self._base_net_vectorized
		out_chann = net_opts['hid_layer_width']
		for lay in range(net_opts['num_hid_layers']):
			with tf.variable_scope('fc_%d' % lay) as scope:			
				activation = fc_layer(in_feat,out_chann,net_opts,False)
				in_feat = activation
				
		out_chann = num_labels
		with tf.variable_scope('fc_final') as scope:

			activation = fc_layer(in_feat,out_chann,net_opts,True)
			
			if net_opts['is_target_distribution']:
				activation = tf.nn.relu(activation, name='activation')
				activation = activation/tf.reduce_sum(activation,-1)
			else: #assumed we have binary target if not distribution
				activation = tf.nn.sigmoid(activation)
					
			return activation
		
	def load_weights(self,init_weight_fname,sess):
		"Initializes (or re-initializes) the pre-trained base model of the network from file. Should be called after running global initializer and before using."
		self._base_net.load_weights(init_weight_fname,sess)

	def _weighted_prior_loss(self,net_opts):
		"specific loss we may want to use in the case of a binary prior."
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
			target = tf.reduce_mean(target,[1,2])
		else:
			target = tf.reduce_max(target,[1,2])
		return target
		
	#expands a 2d map of the correct classes for a tensor and expands to 3d one-hot vectors
	#tensors go from 3d to 4d--batch dim at beginning
	def _expand_target(self,target,num_labels): 
		target = tf.one_hot(target,num_labels+1,dtype=DAT_TYPE,axis=3)
		target = target[:,:,:,1:]
		return target
		
	