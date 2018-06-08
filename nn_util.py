"Common substructures for your neural net--fc layer with all the bells and whisltes, etc."
import tensorflow as tf
import numpy as np

#TODO conv layer
DAT_TYPE = tf.float32

#TODO add batch norm updates to fresh collection by getting current scope--useful for reducing coupling in future
def fc_layer(in_feat,out_chann,net_opts,is_train,is_last_layer=False):
	in_chann = in_feat.shape.as_list()[-1]
	weights = tf.get_variable('weights',[in_chann,out_chann],DAT_TYPE,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight']))
	biases = tf.Variable(tf.constant(0.01, shape=[out_chann], dtype=DAT_TYPE),
			 trainable=True, name='biases')
		
	activation = tf.matmul(in_feat,weights)
	activation = tf.nn.bias_add(activation, biases)
	
	#you may not wish to relu on final layer
	if not is_last_layer:
		activation = tf.nn.relu(activation, name='activation')
		if net_opts['is_fc_batchnorm']:
			activation = tf.layers.batch_normalization(activation,training=is_train,name='fcbn',renorm=True)
		activation = tf.nn.dropout(activation,net_opts['dropout_prob'])	
			
	#we use this because all the weights WE create are added to this collection so they can be trained on their own.
	tf.add_to_collection('fresh',weights)
	tf.add_to_collection('fresh',biases)
	
	return activation
	
	
def pool_score_map_to_prior(target,net_opts):
	"downsamples the processed 4d target tensor to form a prior. Can form binary or histogram prior"
	if net_opts['is_target_distribution']:
		target = tf.reduce_mean(target,[1,2])
	else:
		target = tf.reduce_max(target,[1,2])
	return target
	

def expand_target_2d_to_3d(target,num_labels): 
	"#expands a 2d map of the correct classes for a tensor and expands to 3d one-hot vectors by which I mean tensors go from 3d to 4d--batch dim is at beginning"
	target = tf.one_hot(target,num_labels+1,dtype=DAT_TYPE,axis=3)
	target = target[:,:,:,1:]
	return target