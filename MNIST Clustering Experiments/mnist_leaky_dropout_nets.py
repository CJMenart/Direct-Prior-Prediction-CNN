#this is an experiment designed to run two networks on cleanly divided clusters of the
#MNIST dataset, to see if running nets on a portion of data allows you to learn that
#data better. We use the variance when evaluating with dropout to decide which networks
#to use. We also show each network 'a few' pieces of data from outside its cluster. BTW this is patently insane.

from loss_functions import *
import numpy as np
import random
from print_to_file import *
import os

NCLASS = 10
BATCH_SIZE = 32
WIDTHS = [32,64]
FC_WIDTH = 512
VAR_POP = 16
DROP_PROB = 0.5
INDEX_LISTS = {}
CLASS_SETS = {}
CLASS_SETS['ALL'] = [0,1,2,3,4,5,6,7,8,9]
CLASS_SETS['ROUND'] = [0,2,3,6,8]
CLASS_SETS['STRAIGHT'] = [1,4,5,7,9]
ITER_PER_EVAL = 1000
CLASS_NET_TYPES = ['ALL','ROUND','STRAIGHT']
LEAK_CHANCE = 0.32  #make sure is less than 50%...

def mnist_expr(gpu=None):

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
	
	if gpu is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
	sess = tf.InteractiveSession()
	
	#create simple models	
	nets = {}
	inputs = {}
	truths = {}
	losses = {}
	variance = {}
	
	is_train = tf.placeholder(tf.bool,None)
	total_loss = 0
	for type in CLASS_NET_TYPES:
		inputs[type] = tf.placeholder(tf.float32,[None,28,28,1])
		truths[type] = tf.placeholder(tf.int64,[None])
		INDEX_LISTS[type] = []
		with tf.variable_scope(type):
			feat = net_architecture(inputs[type],is_train)
			final_weights = tf.get_variable('final_weights',[FC_WIDTH,NCLASS],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/FC_WIDTH)))
			final_bias = tf.get_variable('final_bias',[NCLASS],tf.float32,initializer=tf.constant_initializer(0.01))
			feat = tf.matmul(feat,final_weights)
			feat = tf.nn.bias_add(feat,final_bias)			
			nets[type] = tf.nn.softmax(feat,-1)
			
			losses[type] = categorical_cross_entropy_loss(nets[type],tf.one_hot(truths[type],NCLASS),1e-8)
			total_loss += losses[type]
			_,variance[type] = tf.nn.moments(nets[type],axes=[0])
				
	train = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	train = tf.group(train,*update_ops)
	tf.global_variables_initializer().run()

	for iter in range(40001):
		feed_dict = {is_train:True}
		for type in CLASS_NET_TYPES:
			feed_dict[inputs[type]],feed_dict[truths[type]] = get_batch(train_data,train_labels,type)
		sess.run(train,feed_dict=feed_dict)
		if iter % ITER_PER_EVAL == 0:
			print(iter)
			eval(nets,eval_data,eval_labels,inputs,truths,is_train,variance,sess)
		

def get_batch(data,labels,type):
	"get batch with correct restrictions"
	
	#allowed_inds = np.array([i for i in range(len(labels)) if labels[i] in class_set])
	inds = []
	for i in range(BATCH_SIZE):
		if len(INDEX_LISTS[type]) == 0:
			INDEX_LISTS[type] = [i for i in range(len(labels)) if (labels[i] in CLASS_SETS[type] or np.random.uniform() < LEAK_CHANCE)]
			print(type + ' refilling')
			print(len(INDEX_LISTS[type]))
		inds.append(INDEX_LISTS[type].pop(random.randint(0,len(INDEX_LISTS[type])-1)))
	
	truths = labels[inds]
	return(data[inds,:,:,:],truths)

	
def eval(nets,eval_data,eval_labels,inputs,truths,is_train,variance,sess):
	acc_counts = {}
	in_spread_record = {}
	out_spread_record = {}
	
	for type in CLASS_NET_TYPES:
		acc_counts[type] = []
		in_spread_record[type] = []
		out_spread_record[type] = []
	acc_counts['CLUST'] = []
	acc_counts['SYS'] = []
	
	for item in range(len(eval_labels)):
		
		feed_dict = {is_train:False}		
		for type in CLASS_NET_TYPES:			
			feed_dict[inputs[type]] = np.tile(eval_data[np.newaxis,item,:,:,:],[VAR_POP,1,1,1])
			feed_dict[truths[type]] = np.tile(eval_labels[np.newaxis,item],[VAR_POP])

		ans = {}
		spreads = {}
		ans['ALL'], ans['ROUND'], ans['STRAIGHT'], spreads['ALL'], spreads['ROUND'], spreads['STRAIGHT'] = sess.run([nets['ALL'],nets['ROUND'],nets['STRAIGHT'], variance['ALL'], variance['ROUND'], variance['STRAIGHT']],feed_dict=feed_dict)
		for type in CLASS_NET_TYPES:
			ans[type] = np.mean(ans[type],axis=0)
		for type in CLASS_NET_TYPES:
			if eval_labels[item] in CLASS_SETS[type]:
				acc_counts[type].append(np.argmax(ans[type],-1)==eval_labels[item])
				in_spread_record[type].extend(spreads[type])
			else:
				out_spread_record[type].extend(spreads[type])
		clust_assignment = np.mean(spreads['ROUND'])/(np.mean(spreads['STRAIGHT'])+np.mean(spreads['ROUND'])+1e-8)
		#print(spreads['STRAIGHT'])
		#print(spreads['ROUND'])
		#print(clust_assignment)
		#print(eval_labels[item] in CLASS_SETS['STRAIGHT'])
		acc_counts['CLUST'].append(np.round(clust_assignment) == (eval_labels[item] in CLASS_SETS['STRAIGHT']))
		acc_counts['SYS'].append(np.argmax((1-clust_assignment)*ans['ROUND'] + clust_assignment*ans['STRAIGHT']) == eval_labels[item])
	
	accs = {}
	in_spreads = {}
	out_spreads = {}
	for type in CLASS_NET_TYPES+['CLUST','SYS']:
		accs[type] = np.mean(acc_counts[type])
		print('Net %s: acc %.4f' % (type,accs[type]))
	for type in CLASS_NET_TYPES:
		in_spreads[type] = np.mean(in_spread_record[type])
		out_spreads[type] = np.mean(out_spread_record[type])
		print('Net %s: in-variance %.4f, out-variance %.4f' % (type,in_spreads[type],out_spreads[type]))
	file_print(','.join(["%.5f"%accs[type] for type in CLASS_NET_TYPES+['CLUST','SYS']]),'leaky_dropout_%.3f_err.csv' % LEAK_CHANCE)
	file_print(','.join(["%.5f"%in_spreads[type] for type in CLASS_NET_TYPES]) + ','.join(["%.5f"%out_spreads[type] for type in CLASS_NET_TYPES]),'leaky_dropout_%.3f_variance.csv' % LEAK_CHANCE)

	
def clust_assignment_target(truth):
	return int(truth in CLASS_SETS['STRAIGHT'])

	
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