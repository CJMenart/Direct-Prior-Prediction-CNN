"Wraps around a base network (ResNet or Inception) and predicts a distribution directly from passed-in image"

import tensorflow as tf
from activation_summary import *
import os
import numpy as np
from loss_functions import *
import prob_remapping
from base_fcn import *
from nn_util import *
import warnings

slim = tf.contrib.slim #slim is used by the official TF implementation of ResNet
DAT_TYPE = tf.float32
DEBUG = True

#TODO: Make nets inherit from a 'net' class with certain public things exposed?


class PriorNet:

	def __init__(self,net_opts,num_labels,inputs,seg_target,is_train,class_frequency=None,remap_target=None,remap_base_prob=None,map_mat=None):
		
		# we are loading segmentation maps and computing targets from them in case we change this later to be more advanced. Future-proofing. Shouldn't be a performance hit
		#remap_target, remap_base_prob, map_mat are not necessary to feed if you don't use remapping loss or run seg_acc
		
		self.processed_seg_target = expand_target_2d_to_3d(seg_target,num_labels)
		self.prior_target = pool_score_map_to_prior(self.processed_seg_target,net_opts)	
		self.class_frequency = class_frequency
		
		self._base_net = BaseFCN(net_opts,inputs,is_train)
		#TODO: consider changing the type of pooling? Max pooling or something? I'd concat both but too many params
		self._base_net_vectorized = tf.reduce_max(self._base_net.out,[1,2])
		
		activation_summary(self._base_net_vectorized,'base_net_vectorized')

		self.prior = self._prior_predictor(net_opts,num_labels)
		self.direct_prior_loss = self._prior_loss(net_opts)
		
		#What's interesting is that if we didn't have to save and load softmax, but were performing remapping using a model in tensorflow that we ran, this memory limit
		#wouldn't really exist. When we're loading off disk, we have to use 100-pixel examples, but we could remap the whole thing with an integrated model, it's much faster.
		#but right now we're refining a matconvnet model...still, keep this in mind for future
		if net_opts['remapping_loss_weight'] > 0:
			self.remapping_loss,self.seg_acc,_ = prob_remapping.conf_remap_loss_and_metrics(map_mat,self.prior,remap_base_prob,remap_target,net_opts['batch_size'],net_opts['epsilon'])
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
	
	def load_weights(self,init_weight_fname,sess):
		"Initializes (or re-initializes) the pre-trained base model of the network from file. Should be called after running global initializer and before using."
		self._base_net.load_weights(init_weight_fname,sess)
		
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

	def _prior_loss(self,net_opts):
		"specific loss we may want to use in the case of a binary prior."
		if net_opts['is_target_distribution']:
			if net_opts['is_loss_weighted_by_class']:
				warnings.warn('full cross-entropy loss on distribution should not/will not be weighted by class frequency.')
			#TODO options for various loss
			return kl_divergence_loss(self.prior,self.prior_target,net_opts['epsilon'])
		else:
			#weighted cross-entropy loss. OR not-weighted cross-entropy loss
			if net_opts['is_loss_weighted_by_class']:
				return weighted_cross_entropy_loss(self.prior,self.prior_target,self.class_frequency,net_opts['epsilon'])
			else:
				return cross_entropy_loss(self.prior,self.prior_target,net_opts['epsilon'])
				