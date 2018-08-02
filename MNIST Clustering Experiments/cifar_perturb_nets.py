#How well does bagging perform with these networks on MNIST? For use in comparison with self_assign cluster nets

from loss_functions import *
import numpy as np
import random
from print_to_file import *
import os
from nn_util import *
import pickle as pkl

NCLASS = 10
BATCH_SIZE = 64
DROP_PROB = 0.0
CONV_WIDTH = [16,32,64]
FC_WIDTH = 256
CUR_INDEX_LISTS = [[]]
ITER_PER_EVAL = 1000
NET_OPTS = {'perturb_range':0.1,'regularization_weight':1e-4,'is_fc_batchnorm':True,'dropout_prob':DROP_PROB}

def cifar_expr(gpu=None):

	#load data
	data_dir = r'C:\Users\menart.1\Documents\Classification Datasets\cifar-10'
	train_data = None
	train_labels = None
	for batch in range(5):
		with open(os.path.join(data_dir,'data_batch_%d' % (batch+1)),'rb') as f:
			batch = pkl.load(f,encoding='bytes')
		data = np.reshape(np.transpose(batch[b'data']),[32,32,3,-1])
		if train_data is None:
			train_data = data
			train_labels = batch[b'labels']
		else:
			train_data = np.concatenate((train_data,data),-1)
			train_labels = np.concatenate((train_labels,batch[b'labels']),-1)
	with open(os.path.join(data_dir,'test_batch'),'rb') as f:
			test_batch = pkl.load(f,encoding='bytes')
	eval_data = np.reshape(np.transpose(test_batch[b'data']),[32,32,3,-1])
	eval_labels = np.array(test_batch[b'labels'])
	train_data = np.moveaxis(train_data,-1,0)
	eval_data = np.moveaxis(eval_data,-1,0)
	
	r'''
	#load data
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
	print('len train labels')
	print(len(train_labels))
	
	#shrink test set for runtime. Every other. 5000 items.
	eval_data = eval_data[::2,:]
	eval_labels = eval_labels[::2]
	train_data = np.reshape(train_data,[-1,28,28,1])
	eval_data = np.reshape(eval_data,[-1,28,28,1])
	'''
	
	if gpu is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
	sess = tf.InteractiveSession()
	
	#create simple models	
	nets = []
	losses = []
	total_loss = 0
	inputs = tf.placeholder(tf.float32,[None] + list(train_data.shape[1:]))
	truths = tf.placeholder(tf.int64,[None])
	expanded_truths = tf.one_hot(truths,NCLASS)
	is_train = tf.placeholder(tf.bool,None)
	
	with tf.variable_scope('normal'):
		feat_normal = net_architecture(inputs,is_train,False)
		nets.append(tf.nn.softmax(fc_layer(feat_normal,NCLASS,NET_OPTS,is_train,True),-1))

	with tf.variable_scope('perturbed'):
		feat_perturbed = net_architecture(inputs,is_train,True)
		nets.append(tf.nn.softmax(fc_layer(feat_perturbed,NCLASS,NET_OPTS,is_train,True),-1))
		
	with tf.variable_scope('squat_perturbed'):
		feat_squat = squat_net_architecture(inputs,is_train,True)
		nets.append(tf.nn.softmax(fc_layer(feat_squat,NCLASS,NET_OPTS,is_train,True),-1))
	
	for net in range(len(nets)):
		losses.append(cross_entropy_loss(nets[net], expanded_truths,1e-8))
		total_loss += losses[net]
			
	train = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	train = tf.group(train,*update_ops)
	tf.global_variables_initializer().run()

	for iter in range(100001):
		feed_dict = {is_train:True}
		feed_dict[inputs],feed_dict[truths] = get_batch(train_data,train_labels)
		sess.run(train,feed_dict=feed_dict)
		if iter % ITER_PER_EVAL == 0:
			print(iter)
			eval(nets,eval_data,eval_labels,inputs,truths,is_train,sess)
		

def get_batch(data,labels):
	"get batch with correct restrictions"
	
	inds = []
	#allowed_inds = np.array([i for i in range(len(labels)) if labels[i] in class_set])
	for i in range(BATCH_SIZE):
		if len(CUR_INDEX_LISTS[0]) == 0:
			CUR_INDEX_LISTS[0] = list(range(len(labels)))
		inds.append(CUR_INDEX_LISTS[0].pop(random.randint(0,len(CUR_INDEX_LISTS[0])-1)))
	
	return(data[inds,:,:,:],labels[inds])


def eval(nets,eval_data,eval_labels,inputs,truths,is_train,sess):
	acc_counts = [[] for i in range(len(nets))]
	accs = [-1 for i in range(len(nets))]
	
	for item in range(len(eval_labels)):
		feed_dict = {is_train:False}		
		for net in range(len(nets)):
			feed_dict[inputs] = eval_data[np.newaxis,item,:,:,:]
			feed_dict[truths] = eval_labels[np.newaxis,item]

		ans = sess.run(nets,feed_dict=feed_dict)
		for net in range(len(nets)):
			acc_counts[net].append(np.argmax(ans[net]) == eval_labels[item])
					
	for net in range(len(nets)):
		accs[net] = np.mean(acc_counts[net])
		print('Net %d: Acc %.4f' %(net,accs[net]))
	
	file_print(','.join(["%.5f"%accs[net] for net in range(len(nets))]),'perturb_err.csv')

	
def net_architecture(inputs,is_train,is_perturbed):	
	in_chann = 1
	feat = inputs                                         
	for block in range(3):
		for lay in range(2):
			with tf.variable_scope('%d_%d' % (block,lay)):
				if is_perturbed:
					feat = perturbative_layer(feat,CONV_WIDTH[block],NET_OPTS,is_train)
				else:
					feat = conv_layer(feat,CONV_WIDTH[block],NET_OPTS,is_train)
			
		feat = tf.nn.max_pool(feat,[1,2,2,1],[1,2,2,1],'SAME')

	with tf.variable_scope('fc_hidden'):
		feat = tf.contrib.layers.flatten(feat)
		feat = fc_layer(feat,FC_WIDTH,NET_OPTS,is_train)
	
	return feat

	
def squat_net_architecture(inputs,is_train,is_perturbed):	
	in_chann = 1
	feat = inputs                                         
	for block in range(2):
		for lay in range(1):
			with tf.variable_scope('%d_%d' % (block,lay)):
				if is_perturbed:
					feat = perturbative_layer(feat,CONV_WIDTH[block],NET_OPTS,is_train)
				else:
					feat = conv_layer(feat,CONV_WIDTH[block],NET_OPTS,is_train)
			
		feat = tf.nn.max_pool(feat,[1,2,2,1],[1,2,2,1],'SAME')
	feat = tf.nn.max_pool(feat,[1,2,2,1],[1,2,2,1],'SAME')
		
	with tf.variable_scope('fc_hidden'):
		feat = tf.contrib.layers.flatten(feat)
		feat = fc_layer(feat,FC_WIDTH,NET_OPTS,is_train)
	
	return feat

