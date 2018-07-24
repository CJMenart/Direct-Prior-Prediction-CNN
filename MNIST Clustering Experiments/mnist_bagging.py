#How well does bagging perform with these networks on MNIST? For use in comparison with self_assign cluster nets

from loss_functions import *
import numpy as np
import random
from print_to_file import *
import os

NCLASS = 10
BATCH_SIZE = 32
DROP_PROB = 0.0
NUM_BAG = 2
CONV_WIDTH = 32
FC_WIDTH = 256
INDEX_LISTS = []
CUR_INDEX_LISTS = []
ITER_PER_EVAL = 1000
TRAIN_FRAC_PER_BAG = 0.75

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
	nets = []
	losses = []
	inputs = []
	truths = []
	
	is_train = tf.placeholder(tf.bool,None)
	total_loss = 0
	
	for net in range(NUM_BAG):
		with tf.variable_scope(str(net)):
		
			inputs.append(tf.placeholder(tf.float32,[None,28,28,1]))
			truths.append(tf.placeholder(tf.int64,[None]))
			expanded_truths = tf.one_hot(truths[net],NCLASS)
			feat = net_architecture(inputs[net],is_train,net)
			final_weights = tf.get_variable('final_weights',[FC_WIDTH,NCLASS],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/FC_WIDTH)))
			final_bias = tf.get_variable('final_bias',[NCLASS],tf.float32,initializer=tf.constant_initializer(0.01))
			feat = tf.matmul(feat,final_weights)
			feat = tf.nn.bias_add(feat,final_bias)
			nets.append(tf.nn.softmax(feat,-1))
			
			losses.append(cross_entropy_loss(nets[net], expanded_truths,1e-8))
			total_loss += losses[net]
	
	final_bag = tf.add_n(nets)
	final_bag = final_bag/NUM_BAG
		
	train = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	train = tf.group(train,*update_ops)
	tf.global_variables_initializer().run()

	for iter in range(100001):
		feed_dict = {is_train:True}
		for net in range(NUM_BAG):
			feed_dict[inputs[net]],feed_dict[truths[net]] = get_batch(train_data,train_labels,net)
		sess.run(train,feed_dict=feed_dict)
		if iter % ITER_PER_EVAL == 0:
			print(iter)
			eval(nets,final_bag,eval_data,eval_labels,inputs,truths,is_train,sess)
		

def get_batch(data,labels,net):
	"get batch with correct restrictions"
	
	#allowed_inds = np.array([i for i in range(len(labels)) if labels[i] in class_set])
	if len(INDEX_LISTS) < net+1:
		INDEX_LISTS.append([])
		CUR_INDEX_LISTS.append([])
		INDEX_LISTS[net] = np.random.permutation(len(labels))[:int(len(labels)*TRAIN_FRAC_PER_BAG)]
	inds = []
	for i in range(BATCH_SIZE):
		if len(CUR_INDEX_LISTS[net]) == 0:
			CUR_INDEX_LISTS[net] = list(INDEX_LISTS[net])
		inds.append(CUR_INDEX_LISTS[net].pop(random.randint(0,len(CUR_INDEX_LISTS[net])-1)))
	
	return(data[inds,:,:,:],labels[inds])

	
def eval(nets,final_bag,eval_data,eval_labels,inputs,truths,is_train,sess):
	acc_counts = [[] for i in range(NUM_BAG)]
	accs = [[] for i in range(NUM_BAG)]
	final_acc_count = []
	
	for item in range(len(eval_labels)):
		feed_dict = {is_train:False}		
		for net in range(NUM_BAG):
			feed_dict[inputs[net]] = eval_data[np.newaxis,item,:,:,:]
			feed_dict[truths[net]] = eval_labels[np.newaxis,item]

		ans = sess.run(nets + [final_bag],feed_dict=feed_dict)
		for net in range(len(nets)):
			acc_counts[net].append(np.argmax(ans[net]) == eval_labels[item])
		final_acc_count.append(np.argmax(ans[-1]) == eval_labels[item])
		final_acc = np.mean(final_acc_count)
					
	for net in range(len(nets)):
		accs[net] = np.mean(acc_counts[net])
		print('Net %d: Acc %.4f' %(net,accs[net]))
	print('Final Bag: Acc %.4f' % final_acc)
	
	file_print(','.join(["%.5f"%accs[net] for net in range(len(nets))] + ['%.5f' % final_acc]),'bagging_err.csv')
	
	
def net_architecture(inputs,is_train,net):	
	in_chann = 1
	feat = inputs                                         
	for block in range(2):
		for lay in range(3):
			out_chann = CONV_WIDTH*(block+1)
			conv_weights = tf.get_variable('conv%d_%d_weights' % (block,lay),[3,3,in_chann,out_chann],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/9*in_chann))) # regularizer? Eh.
			bias = tf.get_variable('conv%d_%d_bias' % (block,lay),[out_chann],tf.float32,initializer=tf.constant_initializer(0.01))
			feat = tf.nn.conv2d(feat,conv_weights,[1,1,1,1],'SAME')
			feat = tf.nn.bias_add(feat,bias)
			feat = tf.nn.relu(feat)
			feat = tf.layers.batch_normalization(feat,training=is_train,renorm=True)
			in_chann = out_chann
				
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
	
	return feat
