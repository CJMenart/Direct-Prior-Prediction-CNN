#this is an experiment designed to run two networks on cleanly divided clusters of the
#MNIST dataset, to see if running nets on a portion of data allows you to learn that
#data better.
from loss_functions import *
import numpy as np
import random
from print_to_file import *

NCLASS = 10
BATCH_SIZE = 16
WIDTH = 128
DROP_PROB = 0
INDEX_LISTS = {}
CLASS_SETS = {}
CLASS_SETS['ALL'] = [0,1,2,3,4,5,6,7,8,9]
CLASS_SETS['ROUND'] = [0,2,3,6,8]
CLASS_SETS['STRAIGHT'] = [1,4,5,7,9]
CLASS_SETS['CLUST'] = CLASS_SETS['ALL']
ITER_PER_EVAL = 500
CLASS_NET_TYPES = ['ALL','ROUND','STRAIGHT']
ALL_NET_TYPES = CLASS_NET_TYPES + ['CLUST']

def mnist_expr():

	#load data
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
	train_data = np.reshape(train_data,[-1,28,28,1])
	eval_data = np.reshape(eval_data,[-1,28,28,1])
	
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
			final_weights = tf.get_variable('final_weights',[WIDTH,NCLASS],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/WIDTH)))
			final_bias = tf.get_variable('final_bias',[NCLASS],tf.float32,initializer=tf.constant_initializer(0.01))
			feat = tf.matmul(feat,final_weights)
			feat = tf.nn.bias_add(feat,final_bias)			
			nets[type] = tf.nn.softmax(feat,-1)
			
			losses[type] = categorical_cross_entropy_loss(nets[type],tf.one_hot(truths[type],NCLASS),1e-8)
			total_loss += losses[type]
			_,variance[type] = tf.nn.moments(nets[type],axes=[0])
		
	inputs['CLUST'] = tf.placeholder(tf.float32,[None,28,28,1])
	truths['CLUST'] = tf.placeholder(tf.int64,[None])
	INDEX_LISTS['CLUST'] = []
	with tf.variable_scope('CLUST'):
		feat = net_architecture(inputs['CLUST'],is_train)
		final_weights = tf.get_variable('final_weights',[WIDTH,2],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/WIDTH)))
		final_bias = tf.get_variable('final_bias',[2],tf.float32,initializer=tf.constant_initializer(0.01))
		feat = tf.matmul(feat,final_weights)
		feat = tf.nn.bias_add(feat,final_bias)			
		nets['CLUST'] = tf.nn.softmax(feat,-1)
	
		losses['CLUST'] = categorical_cross_entropy_loss(nets['CLUST'],tf.one_hot(truths['CLUST'],2),1e-8)
		total_loss += losses['CLUST']
		#_,variance[type] = tf.nn.moments(nets[type],axes=[0])
		
	train = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	train = tf.group(train,*update_ops)
	tf.global_variables_initializer().run()

	for iter in range(20000):
		feed_dict = {is_train:True}
		for type in ALL_NET_TYPES:
			feed_dict[inputs[type]],feed_dict[truths[type]] = get_batch(train_data,train_labels,BATCH_SIZE,type)
		sess.run(train,feed_dict=feed_dict)
		if iter % ITER_PER_EVAL == 0:
			print(iter)
			eval(CLASS_NET_TYPES,nets,eval_data,eval_labels,inputs,truths,is_train,sess)
		

def get_batch(data,labels,batch_size,type):
	"get batch with correct restrictions"
	
	#allowed_inds = np.array([i for i in range(len(labels)) if labels[i] in class_set])
	inds = []
	for i in range(batch_size):
		if len(INDEX_LISTS[type]) == 0:
			INDEX_LISTS[type] = [i for i in range(len(labels)) if labels[i] in CLASS_SETS[type]]
		inds.append(INDEX_LISTS[type].pop(random.randint(0,len(INDEX_LISTS[type])-1)))
	
	#shuffle = np.random.permutation(len(allowed_inds))[:batch_size]
	#print(shuffle)
	#inds = allowed_inds[shuffle]
	truths = labels[inds]
	if type == 'CLUST':
		truths = [clust_assignment_target(label) for label in truths]
	return(data[inds,:,:,:],truths)

	
def eval(CLASS_NET_TYPES,nets,eval_data,eval_labels,inputs,truths,is_train,sess):
	acc_counts = {}
	
	for type in ALL_NET_TYPES:
		acc_counts[type] = []
	acc_counts['SYS'] = []
	
	for item in range(len(eval_labels)):
		
		feed_dict = {is_train:False}		
		for type in CLASS_NET_TYPES:			
			feed_dict[inputs[type]] = eval_data[np.newaxis,item,:,:,:]
			feed_dict[truths[type]] = eval_labels[np.newaxis,item]
		feed_dict[inputs['CLUST']] = eval_data[np.newaxis,item,:,:,:]
		feed_dict[truths['CLUST']] = [clust_assignment_target(eval_labels[item])]
		#print(feed_dict[truths['CLUST']])

		ans = {}
		ans['ALL'], ans['ROUND'], ans['STRAIGHT'], ans_clust = sess.run([nets['ALL'],nets['ROUND'],nets['STRAIGHT'],nets['CLUST']],feed_dict=feed_dict)
		for type in CLASS_NET_TYPES:
			if eval_labels[item] in CLASS_SETS[type]:
				acc_counts[type].append(np.argmax(ans[type],-1)==eval_labels[item])
		acc_counts['CLUST'].append(np.argmax(ans_clust) == feed_dict[truths['CLUST']])
		acc_counts['SYS'].append(np.argmax(ans_clust[:,0]*ans['ROUND'] + ans_clust[:,1]*ans['STRAIGHT']) == eval_labels[item])
	
	accs = {}
	for type in ALL_NET_TYPES+['SYS']:
		accs[type] = np.mean(acc_counts[type])
		print('Net %s: acc %.4f' % (type,accs[type]))
	file_print(','.join(["%.5f"%accs[type] for type in ALL_NET_TYPES+['SYS']]),'err.csv')
	
	
def clust_assignment_target(truth):
	return int(truth in CLASS_SETS['STRAIGHT'])

	
def net_architecture(inputs,is_train):	
	in_chann = 1
	out_chann = WIDTH
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
			out_chann = WIDTH
				
		feat = tf.nn.max_pool(feat,[1,2,2,1],[1,2,2,1],'SAME')
	
	#feat = tf.reduce_mean(feat,axis=[1,2])
	feat = tf.contrib.layers.flatten(feat)
	feat = tf.nn.dropout(feat,1-DROP_PROB)
	in_chann = feat.shape.as_list()[-1]
	
	fc_weights = tf.get_variable('fc_weights',[in_chann,WIDTH],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)))
	fc_bias = tf.get_variable('fc_bias',[WIDTH],tf.float32,initializer=tf.constant_initializer(0.01))
	feat = tf.matmul(feat,fc_weights)
	feat = tf.nn.bias_add(feat,fc_bias)
	feat = tf.nn.relu(feat)
	feat = tf.layers.batch_normalization(feat,training=is_train,renorm=True)
	in_chann = WIDTH
	
	return feat
	
'''
def eval_dropout_something_something()

	#test
	acc_counts = {}
	in_spread_record = {}
	out_spread_record = {}
	for type in CLASS_NET_TYPES:
		acc_counts[type] = []
		in_spread_record[type] = []
		out_spread_record[type] = []
	for item in range(len(eval_labels)):
				
		for type in CLASS_NET_TYPES:			
			feed_dict = {is_train:False}
			#feed_dict[inputs[type]] = np.tile(eval_data[np.newaxis,item,:,:,:],[32,1,1,1])
			#feed_dict[truths[type]] = np.tile(eval_labels[np.newaxis,item],[32])
			feed_dict[inputs[type]] = eval_data[np.newaxis,item,:,:,:]
			feed_dict[truths[type]] = eval_labels[np.newaxis,item]
			
			if eval_labels[item] in CLASS_SETS[type]:
				ans,spread = sess.run([nets[type],variance[type]],feed_dict=feed_dict)
				#print(ans)
				#print(eval_labels[item])
				#print(np.argmax(ans)==eval_labels[item])
				print(spread)
				acc_counts[type].append(np.argmax(ans,-1)==eval_labels[item])
				in_spread_record[type].extend(spread)
			else:
				ans,spread = sess.run([nets[type],variance[type]],feed_dict=feed_dict)
				out_spread_record[type].extend(spread)
'''