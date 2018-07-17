#this is an experiment designed to run multiple networks with/without SPARSITY in the connections, to see if we can reduce the number of weights via a
#trivial mechanism without degrading performance.

#TODO: for a 'paper-quality' trial, do multiple networks simultaneously, dense and sparse, at each trial,
#and also record training accuracy

from loss_functions import *
import numpy as np
import random
from print_to_file import *
from nn_util import *
import os

NCLASS = 10
BATCH_SIZE = 32
W = 24
WIDTHS = [W,2*W]
FC_WIDTH = 6*W
DROP_PROB = 0
ITER_PER_EVAL = 500
INDEX_LIST = [[]]
NET_TYPES=['DENSE','SPARSE']

def mnist_expr(gpu=None):

	#load data
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
	train_data = np.reshape(train_data,[-1,28,28,1])
	eval_data = np.reshape(eval_data,[-1,28,28,1])
	
	if gpu is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
	sess = tf.InteractiveSession()
	
	#create simple models	
	nets = {}
	losses = {}
	
	is_train = tf.placeholder(tf.bool,None)
	inputs = tf.placeholder(tf.float32,[None,28,28,1])
	truths = tf.placeholder(tf.int64,[None])
	total_loss = 0
	
	for type in NET_TYPES:		
		with tf.variable_scope(type):
			feat = net_architecture(inputs,is_train,type=='SPARSE')
			final = fc_layer(feat,NCLASS,{'regularization_weight':0.0,'is_fc_batchnorm':True,'dropout_prob':DROP_PROB},is_train,is_last_layer=True)
			nets[type] = tf.nn.softmax(final,-1)
			losses[type] = categorical_cross_entropy_loss(nets[type],tf.one_hot(truths,NCLASS),1e-8)
			total_loss += losses[type]
				
	train = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	train = tf.group(train,*update_ops)
	tf.global_variables_initializer().run()

	for iter in range(20001):
		feed_dict = {is_train:True}
		feed_dict[inputs],feed_dict[truths] = get_batch(train_data,train_labels,BATCH_SIZE)
		sess.run(train,feed_dict=feed_dict)
		if iter % ITER_PER_EVAL == 0:
			print(iter)
			eval(nets,eval_data,eval_labels,inputs,truths,is_train,sess)
		

def get_batch(data,labels,batch_size):
	"get batch with correct restrictions"
	
	#allowed_inds = np.array([i for i in range(len(labels)) if labels[i] in class_set])
	inds = []
	for i in range(batch_size):
		if len(INDEX_LIST[0]) == 0:
			INDEX_LIST[0] = list(range(len(labels)))
		inds.append(INDEX_LIST[0].pop(random.randint(0,len(INDEX_LIST[0])-1)))
	
	truths = labels[inds]
	return(data[inds,:,:,:],truths)

	
def eval(nets,eval_data,eval_labels,inputs,truths,is_train,sess):
	
	acc_counts = {}
	
	for type in NET_TYPES:
		acc_counts[type] = []
	
	for item in range(len(eval_labels)):
		
		feed_dict = {is_train:False}		
		feed_dict[inputs] = eval_data[np.newaxis,item,:,:,:]
		feed_dict[truths] = eval_labels[np.newaxis,item]

		ans = {}
		ans['DENSE'], ans['SPARSE'] = sess.run([nets['DENSE'],nets['SPARSE']],feed_dict=feed_dict)
		for type in NET_TYPES:
			acc_counts[type].append(np.argmax(ans[type],-1)==eval_labels[item])
		
	accs = {}
	for type in NET_TYPES:
		accs[type] = np.mean(acc_counts[type])
		print('Net %s: acc %.4f' % (type,accs[type]))
	file_print(','.join(["%.5f"%accs[type] for type in NET_TYPES]),'sparse_err_%d.csv' % W)
	
	
def net_architecture(inputs,is_train,is_sparse):	
	in_chann = 1
	feat = inputs                                         
	for block in range(2):
		for lay in range(3):
			with tf.variable_scope('block_%d_lay_%d' % (block,lay)) as scope:
				if is_sparse:
					print('making sparse net')
					feat = sparse_conv_layer(feat,WIDTHS[block],{'regularization_weight':0.0},is_train)				
				else:
					feat = conv_layer(feat,WIDTHS[block],{'regularization_weight':0.0},is_train)
		feat = tf.nn.max_pool(feat,[1,2,2,1],[1,2,2,1],'SAME')
	
	with tf.variable_scope('fc') as scope:
		feat = tf.contrib.layers.flatten(feat)
		if is_sparse:
			feat = sparse_fc_layer(feat,FC_WIDTH,{'regularization_weight':0.0,'is_fc_batchnorm':True,'dropout_prob':DROP_PROB},is_train)
		else:
			feat = fc_layer(feat,FC_WIDTH,{'regularization_weight':0.0,'is_fc_batchnorm':True,'dropout_prob':DROP_PROB},is_train)
	
	return feat
	