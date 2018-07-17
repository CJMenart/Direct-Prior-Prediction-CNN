#After previous experiments, learning separate cluster nets is bad. This script (in progress) is to test learning a series of nets which each separate one class from all others,
#because we're learning all the boundaries, to determine if there is really a benefit from training things separately.

from loss_functions import *
import numpy as np
import random
from print_to_file import *
import os

NCLASS = 10
BATCH_SIZE = 32
WIDTHS = [32,64]
FC_WIDTH = 256
VAR_POP = 16
DROP_PROB = 0.0
INDEX_LIST = [[]]
ITER_PER_EVAL = 1000

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
	one_num_nets = []
	losses = []
	
	is_train = tf.placeholder(tf.bool,None)
	inputs = tf.placeholder(tf.float32,[None,28,28,1])
	truths = tf.placeholder(tf.int64,[None])
	total_loss = 0
	
	for digit in range(NCLASS):
		
		with tf.variable_scope(type):
			feat = net_architecture(inputs[type],is_train)
			final_weights = tf.get_variable('final_weights',[FC_WIDTH,1],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/FC_WIDTH)))
			final_bias = tf.get_variable('final_bias',[1],tf.float32,initializer=tf.constant_initializer(0.01))
			feat = tf.matmul(feat,final_weights)
			feat = tf.nn.bias_add(feat,final_bias)	
			one_num_nets.append(tf.nn.sigmoid(feat,-1))
			
			losses.append(cross_entropy_loss(one_num_nets[digit],tf.equal(truths[digit],digit),1e-8))
			total_loss += losses[digit]

	#'all' net
	with tf.variable_scope('all'):
		feat = net_architecture(inputs,is_train)
		final_weights = tf.get_variable('final_weights',[FC_WIDTH,NCLASS],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/FC_WIDTH)))
		final_bias = tf.get_variable('final_bias',[NCLASS],tf.float32,initializer=tf.constant_initializer(0.01))
		feat = tf.matmul(feat,final_weights)
		feat = tf.nn.bias_add(feat,final_bias)			
		all_net = tf.nn.softmax(feat,-1)
		
		losses.append(categorical_cross_entropy_loss(all_net,tf.one_hot(truths[type],NCLASS),1e-8))
		total_loss += losses[-1]
			
	train = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	train = tf.group(train,*update_ops)
	tf.global_variables_initializer().run()

	for iter in range(20001):
		feed_dict = {is_train:True}
		feed_dict[inputs],feed_dict[truths] = get_batch(train_data,train_labels)
		sess.run(train,feed_dict=feed_dict)
		if iter % ITER_PER_EVAL == 0:
			print(iter)
			eval(one_num_nets,all_net,eval_data,eval_labels,inputs,truths,is_train,sess)
		

def get_batch(data,labels):
	"get batch with correct restrictions"
	
	#allowed_inds = np.array([i for i in range(len(labels)) if labels[i] in class_set])
	inds = []
	for i in range(batch_size):
		if len(INDEX_LISTS[0]) == 0:
			INDEX_LISTS[0] = [i for i in range(len(labels))]
		inds.append(INDEX_LISTS[0].pop(random.randint(0,len(INDEX_LISTS[0])-1)))
	
	truths = labels[inds]
	return(data[inds,:,:,:],truths)

	
def eval(one_num_nets,all_net,eval_data,eval_labels,inputs,truths,is_train,sess):
	acc_num = []*NCLASS
	acc_all = []
	
	for item in range(len(eval_labels)):
		
		feed_dict = {is_train:False}		
		feed_dict[inputs] = eval_data[np.newaxis,item,:,:,:]
		feed_dict[truths] = eval_labels[np.newaxis,item]

		ans = []
		
		**ans, ans_all = sess.run([**ans,acc_all],feed_dict=feed_dict)
		for digit in range(NCLASS):
				acc_num[digit].append(np.round(ans[digit],-1)==(eval_labels[item]==digit))
		acc_all.append(np.argmax(ans_all,-1) == eval_labels[item])

	mean_acc_num = []*NCLASS
	for digit in range(NCLASS):
		mean_acc_num[digit] = np.mean(acc_num[digit])
		print('Net %d: Acc %.4f',%(digit,mean_acc_num[digit])
	mean_acc_all = np.mean(acc_all)
	print('Net ALL: Acc %.4f' % mean_acc_all)
	file_print(','.join(["%.5f"%mean_acc_num[digit] for digit in range(NCLASS)] + ['%.5f'%mean_acc_all]),'err.csv')
	
	
def net_architecture(inputs,is_train):	
	in_chann = 1
	out_chann = WIDTHS[0]
	feat = inputs                                         
	for block in range(2):
		for lay in range(3):
			conv_weights = tf.get_variable('conv%d_%d_weights' % (block,lay),[3,3,in_chann,out_chann],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/9*in_chann))) # regularizer? Eh.
			bias = tf.get_variable('conv%d_%d_bias' % (block,lay),[out_chann],tf.float32,initializer=tf.constant_initializer(0.01))
			feat = tf.nn.conv2d(feat,conv_weights,[1,1,1,1],'SAME')
			feat = tf.nn.bias_add(feat,bias)
			feat = tf.nn.relu(feat)
			feat = tf.layers.batch_normalization(feat,training=is_train,renorm=True)
			in_chann = out_chann
			out_chann = WIDTHS[block]
				
		feat = tf.nn.max_pool(feat,[1,2,2,1],[1,2,2,1],'SAME')
	
	#feat = tf.reduce_mean(feat,axis=[1,2])
	feat = tf.contrib.layers.flatten(feat)
	in_chann = feat.shape.as_list()[-1]
	
	fc_weights = tf.get_variable('fc_weights',[in_chann,FC_WIDTH],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)))
	fc_bias = tf.get_variable('fc_bias',[FC_WIDTH],tf.float32,initializer=tf.constant_initializer(0.01))
	feat = tf.matmul(feat,fc_weights)
	feat = tf.nn.bias_add(feat,fc_bias)
	feat = tf.nn.relu(feat)
	feat = tf.layers.batch_normalization(feat,training=is_train,renorm=True)
	feat = tf.nn.dropout(feat,1-DROP_PROB)
	in_chann = FC_WIDTH
	
	return feat
