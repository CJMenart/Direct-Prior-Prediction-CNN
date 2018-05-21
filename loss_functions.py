import tensorflow as tf

#loss functions here use reduce_mean instead of reduce_sum so that magnitude of loss does not depend on input size
#they are reduce in such a way that categorical and cross-entropy should be 'comparable'

# you can only bother with this if you know your target is 1-hot...
def categorical_cross_entropy_loss(out,truth_vec,epsilon):
	#0 has to be avoided, results in infinite loss. 1 is the 'perfect answer/no loss' so we don't care if we 'saturate' against it
	out = tf.minimum(1.0,out+epsilon)
	loss = tf.reduce_mean(tf.reduce_sum(-1.0*truth_vec*tf.log(out),axis=-1))
	return loss	
	
def cross_entropy_loss(out,truth_vec,epsilon):
	loss = tf.reduce_mean(-1.0*truth_vec*tf.log(tf.minimum(1.0,out+epsilon))) + tf.reduce_mean(-1.0*(1.0-truth_vec)*tf.log(tf.minimum(1.0,1.0-out+epsilon)))
	return loss
	
def segmentation_accuracy(score_map,target,predict0):

	#target is an integer map
	labels = tf.argmax(score_map,axis=3)
	if predict0: #are you trying to predict the '0' class, or does it mean you don't care. i.e. do labels have 0-based indexing
		return tf.reduce_mean(tf.cast(tf.equal(labels, target),tf.float32))   
	else:		
		return tf.reduce_sum(tf.cast(tf.equal(labels+1, target),tf.float32)) /\
				tf.reduce_sum(tf.cast(tf.not_equal(target, 0),tf.float32))   

def segmentation_accuracy_target_3D(score_map,target_3D):
	labels = tf.argmax(score_map,axis=3)
	target = tf.argmax(target_3D,axis=3)
	return tf.reduce_mean(tf.cast(tf.equal(labels, target),tf.float32))   
