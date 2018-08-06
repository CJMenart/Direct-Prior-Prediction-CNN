"various deep-learning loss functions and error measures as tensorflow nodes.\
Many functions take an epsilon, this must be non-zero to prevent possible inf/nan errors."
import tensorflow as tf

#loss functions here use reduce_mean instead of reduce_sum so that magnitude of loss does not depend on input size
#they are reduced in such a way that categorical and cross-entropy should be 'on the same scale'

def categorical_cross_entropy_loss(out,truth_vec,epsilon):
	"simple version of cross-entropy loss for 1-hot target."
	#0 has to be avoided, results in infinite loss. 1 is the 'perfect answer/no loss' so we don't care if we 'saturate' against it
	#but we REALLY SHOULD add epsilon instead of maxxing with epsilon in order to prevent no-gradient from reaching us WHEN out is less than epsilon
	out = tf.minimum(1.0,out+epsilon)
	loss = tf.reduce_mean(tf.reduce_sum(-1.0*truth_vec*tf.log(out),axis=-1))
	return loss	
	
	
def cross_entropy_loss(out,truth_vec,epsilon):
	"most common function for matching 0-1 targets."
	loss = tf.reduce_mean(-1.0*truth_vec*tf.log(tf.minimum(1.0,out+epsilon))) + tf.reduce_mean(-1.0*(1.0-truth_vec)*tf.log(tf.minimum(1.0,1.0-out+epsilon)))
	return loss
	
	
def weighted_cross_entropy_loss(out,truth_vec,class_frequency,epsilon):
	"specific cross-entropy loss we may want to use when the targets in each dimension have different frequencies--if each one represents a class detection and you want them to be balanced to have the same average loss."
	false_pos_weight = class_frequency*2+epsilon
	false_neg_weight = 2-class_frequency*2+epsilon
	if DEBUG:
		print('false_neg_weights:')
		print(false_neg_weight.shape.as_list())
		
	loss = tf.reduce_mean(-false_neg_weight*truth_vec*tf.log(tf.minimum(1.0,out+epsilon))) + tf.reduce_mean(-false_pos_weight*(1-truth_vec)*tf.log(tf.minimum(1.0,1.0-out+epsilon)))
	return loss

	
def kl_divergence_loss(out,truth_vec,epsilon):
	return tf.reduce_mean(tf.reduce_sum(truth_vec*tf.log(truth_vec/tf.minimum(1.0,out+epsilon)+epsilon),axis=-1))
	

def euclidean_distance_loss(out,truth_vec):
	return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(truth_vec-out,2),axis=-1)))

	
def magnitude_diff_loss(out,truth_vec,epsilon):
	"element-wise difference in orders of magnitude between source and target."
	return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.log(out+epsilon)-tf.log(truth_vec+epsilon)),axis=-1))
	
def squared_error_loss(out,truth_vec):
	return tf.reduce_mean(tf.reduce_sum(tf.pow(truth_vec-out,2),axis=-1))

	
def avg_one_inf_norm_loss(out,truth_vec):
	"The average of the L1 and L-Infinity norms of the absolute distance between estimation and target. Researched by\
	Mo Akbar as effective for arbitrary distribution-matching. But I am suspicious of the work, and think this may only be safe with adaptive gradients."
	abs_dist = tf.abs(truth_vec-out)
	return tf.reduce_mean((tf.reduce_sum(abs_dist,axis=-1) + tf.reduce_max(abs_dist,axis=-1) )/2)

	
def chi_squared_loss(out,truth_vec,epsilon):
	"Symmetric version of chi-squared, used for distributions. Ranges 0-2. Mo Akbar also says this one is good."
	return tf.reduce_mean(tf.reduce_sum(tf.pow(out-truth_vec,2)/(out+truth_vec+epsilon),axis=-1))
	
	
def segmentation_accuracy(score_map,target,predict0):
	"determine the accuracy of a semantic segmentation for an image. Set predict0 to true iff you are included the class 0."
	#target is an integer map
	labels = tf.argmax(score_map,axis=3)
	if predict0: #are you trying to predict the '0' class, or does it mean you don't care. i.e. do labels have 0-based indexing
		return tf.reduce_mean(tf.cast(tf.equal(labels, target),tf.float32))   
	else:		
		return tf.reduce_sum(tf.cast(tf.equal(labels+1, target),tf.float32)) /\
				tf.reduce_sum(tf.cast(tf.not_equal(target, 0),tf.float32))   

				
def segmentation_accuracy_target_3D(score_map,target_3D):
	"accuracy of a semantic segmentaion but you can pass in a whole 3d score map of confidence values. Target must be one-hot values."
	labels = tf.argmax(score_map,axis=3)
	target = tf.argmax(target_3D,axis=3)
	return tf.reduce_mean(tf.cast(tf.equal(labels, target),tf.float32))   

	
def hamming_err(out,truth_vec):
	"sum of 'hamming distances', both vectors rounded to be binary"
	rounded = tf.round(out)
	return tf.reduce_sum(tf.abs(rounded-tf.round(truth_vec)))
		
		
def thresh_err(out,truth_vec,thresh):
	"Kind of arbitrary error, number of coordintes whose abs distance exceeds a threshold."
	return tf.reduce_sum(tf.cast(tf.abs(out-truth_vec) >= thresh,tf.float32))
	