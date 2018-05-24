"Common substructures for your neural net--fc layer with all the bells and whisltes, etc."
import tensorflow as tf
import numpy as np

#TODO conv layer
DAT_TYPE = tf.float32

#TODO batch norm? Pull out of net_opts probably
def fc_layer(in_feat,out_chann,net_opts,is_last_layer=False):
	in_chann = in_feat.shape.as_list()[-1]
	weights = tf.get_variable('weights',[in_chann,out_chann],DAT_TYPE,initializer=tf.			truncated_normal_initializer(np.sqrt(2/in_chann)),regularizer=tf.contrib.layers.l2_regularizer(net_opts['regularization_weight']))
	biases = tf.Variable(tf.constant(0.01, shape=[out_chann], dtype=DAT_TYPE),
			 trainable=True, name='biases')
		
	activation = tf.matmul(in_feat,weights)
	activation = tf.nn.bias_add(activation, biases)
	
	#you may not wish to relu on final layer
	if not is_last_layer:
		activation = tf.nn.relu(activation, name='activation')
		activation = tf.nn.dropout(activation,net_opts['dropout_prob'])
			
	#we use this because all the weights WE create are added to this collection so they can be trained on their own.
	tf.add_to_collection('fresh',weights)
	tf.add_to_collection('fresh',biases)
	
	return activation