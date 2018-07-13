"Common substructures for your neural net--fc layer with all the bells and whisltes, etc."
import tensorflow as tf
import numpy as np
from activation_summary import *

#TODO conv layer
DAT_TYPE = tf.float32

#TODO add batch norm updates to fresh collection by getting current scope--useful for reducing coupling in future
def fc_layer(in_feat,out_chann,net_opts,is_train,is_last_layer=False):
	in_chann = in_feat.shape.as_list()[-1]
	weights = tf.get_variable('weights',[in_chann,out_chann],DAT_TYPE,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight']))
	biases = tf.get_variable('biases',[out_chann],DAT_TYPE,initializer=tf.constant_initializer(0.01))
	activation_summary(weights)
	activation_summary(biases)
		
	activation = tf.matmul(in_feat,weights)
	activation = tf.nn.bias_add(activation, biases)
	
	#you may not wish to relu on final layer
	if not is_last_layer:
		activation_summary(activation,'weighted_sum')
		activation = leaky_relu(activation)
		if net_opts['is_fc_batchnorm']:
			activation = tf.layers.batch_normalization(activation,training=is_train,name='fcbn',renorm=True)
		#NOTE dropout will be on at all times under this model b/c of testing spread/stat of data is how we want to do things. Be careful. Consider adding placeholders to control.
		activation = tf.nn.dropout(activation,1-net_opts['dropout_prob'])
			
	activation_summary(activation,'activation')
	#we use this because all the weights WE create are added to this collection so they can be trained on their own.
	tf.add_to_collection('fresh',weights)
	tf.add_to_collection('fresh',biases)
	
	return activation
	

def leaky_relu(x):
	return tf.maximum(x,0.05*x)
	
	
def pool_score_map_to_prior(target,net_opts):
	"downsamples the processed 4d target tensor to form a prior. Can form binary or histogram prior"
	if net_opts['is_target_distribution']:
		target = tf.reduce_mean(target,[1,2])
		target = target/tf.maximum(tf.reduce_sum(target,axis=-1,keep_dims=True),net_opts['epsilon']) #must renorm because unlabeled pixels will result in not-a-full-distribution
	else:
		target = tf.reduce_max(target,[1,2])
	return target
	

def expand_target_2d_to_3d(target,num_labels): 
	"#expands a 2d map of the correct classes for a tensor and expands to 3d one-hot vectors by which I mean tensors go from 3d to 4d--batch dim is at beginning"
	target = tf.one_hot(target,num_labels+1,dtype=DAT_TYPE,axis=3)
	target = target[:,:,:,1:]
	return target
	

def pool_to_fixed(inputs, output_size, mode, vectorize=False):
	"pools a 4d feature map to a feature cube with fixed spatial side lengths, and optionally vectorize."

	inputs_shape = tf.shape(inputs)
	b = tf.cast(tf.gather(inputs_shape, 0), tf.int32)
	h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
	w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)
	#f = tf.cast(tf.gather(inputs_shape, 3), tf.int32)
	#number of feature maps is known ahead of time, so fix
	f = inputs.shape.as_list()[-1]
	
	n = output_size
	result = []
	
	if mode == 'max':
		pooling_op = tf.reduce_max
	elif mode == 'avg':
		pooling_op = tf.reduce_mean
	else:
		msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
		raise ValueError(msg.format(mode))
	
	for row in range(output_size):
		for col in range(output_size):
			start_h = tf.cast(tf.floor(tf.multiply(tf.divide(row, n), tf.cast(h, tf.float32))), tf.int32)
			end_h = tf.cast(tf.ceil(tf.multiply(tf.divide((row + 1), n), tf.cast(h, tf.float32))), tf.int32)
			start_w = tf.cast(tf.floor(tf.multiply(tf.divide(col, n), tf.cast(w, tf.float32))), tf.int32)
			end_w = tf.cast(tf.ceil(tf.multiply(tf.divide((col + 1), n), tf.cast(w, tf.float32))), tf.int32)
			pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
			pool_result = pooling_op(pooling_region, axis=(1, 2))
			result.append(pool_result)
	#print('Pool shape with pool size %d' % output_size)
	#print(result)
	result = tf.concat(result,1)
	#print(result.shape.as_list())
	if not vectorize:
		result = tf.reshape(result,[b,output_size,output_size,f])
		#print(result.shape.as_list())
	return result
	

def pyramid_pool(inputs,pool_sizes,mode,net_opts,shrink_dim=None):
	"Builds a pyramid-pool of a convolutional 4d feature by vecotrizing poolings down to different fixes sizes and concatenating. Similar to PSPSNet."	

	pyramid = None
	in_chann = inputs.shape.as_list()[-1]
	pyramid = pool_to_fixed(inputs, pool_sizes[0], mode, vectorize=True)
	for l in range(0,len(pool_sizes)):
		if pool_sizes[l] is not None: # to make a large pyramid pooling you may need lower dim
			if shrink_dim:
				filter = tf.get_variable('pyramid_weights_%d' % l,[1,1,in_chann,shrink_dim],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight']))
				temp_in = tf.nn.conv2d(inputs,filter,[1,1,1,1],'SAME')
			else:
				temp_in = inputs
			if pyramid is not None:
				pyramid = tf.concat([pyramid,pool_to_fixed(temp_in, pool_sizes[l], mode, vectorize=True)],-1)
			else:
				pyramid = pool_to_fixed(temp_in, pool_sizes[l], mode, vectorize=True)
	return pyramid

	
def grouped_matmul_layer(in_feat,out_chann,group_sz,group_num,net_opts,is_train,is_last_layer=False):
	"Based on 'grouped convolutions' from ResNeXt."
	in_chann = in_feat.shape.as_list()[-1]
	output = None
	
	for group in range(group_num):
		down_weights = tf.get_variable('down_weights_%d' % group,[in_chann,group_sz],DAT_TYPE,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight']))
		reduced = tf.matmul(in_feat,down_weights)
		
		biases = tf.get_variable('biases_%d' % group,[group_sz],DAT_TYPE,initializer=tf.constant_initializer(0.01))
		weights = tf.get_variable('weights_%d' % group,[group_sz,group_sz],DAT_TYPE,initializer=tf.truncated_normal_initializer(np.sqrt(2/group_sz)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight']))
		activation = tf.matmul(reduced,weights)
		activation = tf.nn.bias_add(activation,biases)
		if not is_last_layer:
			activation = leaky_relu(activation)
		
		up_weights = tf.get_variable('up_weights_%d' % group,[group_sz,out_chann],DAT_TYPE,initializer=tf.truncated_normal_initializer(np.sqrt(2/group_sz)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight']))
		expanded = tf.matmul(activation,up_weights)
		if output is not None:
			output = output + expanded
		else:
			output = expanded
		
		#we use this because all the weights WE create are added to this collection so they can be trained on their own.
		tf.add_to_collection('fresh',up_weights)
		tf.add_to_collection('fresh',down_weights)			
		tf.add_to_collection('fresh',weights)
		tf.add_to_collection('fresh',biases)
		
		activation_summary(up_weights)
		activation_summary(down_weights)
		activation_summary(weights)
		activation_summary(biases)
		
	if not is_last_layer:
		if net_opts['is_fc_batchnorm']:
			output = tf.layers.batch_normalization(output,training=is_train,name='fcbn',renorm=True)
		#NOTE dropout will be on at all times under this model b/c of testing spread/stat of data is how we want to do things. Be careful. Consider adding placeholders to control.
		output = tf.nn.dropout(output,1-net_opts['dropout_prob'])
			
	activation_summary(output,'grouped_matmul_output')
	
	return output
	

def sparse_conv_layer(feat,out_chann,net_opts,is_train,is_last_layer=False):
	"Exploring a theme similar to that of grouped convolutions--which are sort of a form of sparse connection--we try\
	sparse convolutions. Filters are sparsely connected to each other--there is no low-rank decomposition. These connections\
	are random."
	in_chann = feat.shape.as_list()[-1]		
	conv_weights = tf.get_variable('conv_weights',[3,3,in_chann,out_chann],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/3*3*in_chann)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight'])) 
	bias = tf.get_variable('conv_bias',[out_chann],tf.float32,initializer=tf.constant_initializer(0.01))
	sparsity_gate_value = np.round(np.random.uniform(size=[1,1,in_chann,out_chann])) #50% keep rate
	sparsity_gate = tf.get_variable('sparsity_gate',[1,1,in_chann,out_chann],tf.int64,initializer=tf.constant_initializer(sparsity_gate_value),trainable=False)
	sparsity_gate_expanded = tf.tile(sparsity_gate,[3,3,1,1])
	
	feat = tf.nn.conv2d(feat,conv_weights*sparsity_gate_expanded,[1,1,1,1],'SAME')
	feat = tf.nn.bias_add(feat,bias)
	if not is_last_layer:
		feat = leaky_relu(feat)
		feat = tf.layers.batch_normalization(feat,training=is_train,renorm=True)
		
	tf.add_to_collection('fresh',conv_weights)
	tf.add_to_collection('fresh',bias)
	
	return feat