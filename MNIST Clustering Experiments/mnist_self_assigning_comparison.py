#After previous experiments, learning separate cluster nets is bad. This script (in progress) is to test learning a series of nets which each separate one class from all others,
#because we're learning all the boundaries, to determine if there is really a benefit from training things separately.
#After this, self-assigning cluster networks will need to be compared to bagging.

from loss_functions import *
import numpy as np
import random
from print_to_file import *
import os

NCLASS = 10
BATCH_SIZE = 32
VAR_POP = 16
DROP_PROB = 0.0
CLASS_LISTS = [[0,2,3,6,8],[1,4,5,7,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]
CONV_WIDTHS = [32,32,32,45,32]
FC_WIDTHS = [256,256,256,362,549]
INDEX_LISTS = []
ITER_PER_EVAL = 1000
LEAK_CHANCE = 0.16

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
	
	for net in range(len(CLASS_LISTS)):
	
		inputs.append(tf.placeholder(tf.float32,[None,28,28,1]))
		truths.append(tf.placeholder(tf.int64,[None]))
		expanded_truths = tf.one_hot(truths[net],NCLASS+1)
		
		with tf.variable_scope(str(net)):
			NOUT = len(CLASS_LISTS[net])+1
			feat = net_architecture(inputs[net],is_train,net)
			final_weights = tf.get_variable('final_weights',[FC_WIDTHS[net],NOUT],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/FC_WIDTHS[net])))
			final_bias = tf.get_variable('final_bias',[NOUT],tf.float32,initializer=tf.constant_initializer(0.01))
			feat = tf.matmul(feat,final_weights)
			feat = tf.nn.bias_add(feat,final_bias)	
			nets.append(tf.nn.softmax(feat,-1))
			
			in_clust_truth = tf.gather(expanded_truths,CLASS_LISTS[net]+[-1],axis=-1)			
			target = tf.tile(tf.expand_dims(tf.one_hot(len(CLASS_LISTS[net]),len(CLASS_LISTS[net])+1),0),[BATCH_SIZE,1])*tf.expand_dims(1-tf.reduce_sum(in_clust_truth,-1),1) + in_clust_truth
			print(target)
			losses.append(cross_entropy_loss(nets[net], target,1e-8))
			total_loss += losses[net]
			
	train = tf.train.AdamOptimizer(1e-2).minimize(total_loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	train = tf.group(train,*update_ops)
	tf.global_variables_initializer().run()

	for iter in range(100001):
		feed_dict = {is_train:True}
		for net in range(len(CLASS_LISTS)):
			feed_dict[inputs[net]],feed_dict[truths[net]] = get_batch(train_data,train_labels,net)
		sess.run(train,feed_dict=feed_dict)
		if iter % ITER_PER_EVAL == 0:
			print(iter)
			eval(nets,eval_data,eval_labels,inputs,truths,is_train,sess)
		

def get_batch(data,labels,net):
	"get batch with correct restrictions"
	
	#allowed_inds = np.array([i for i in range(len(labels)) if labels[i] in class_set])
	inds = []
	if len(INDEX_LISTS) < net+1:
		INDEX_LISTS.append([])
	for i in range(BATCH_SIZE):
		if len(INDEX_LISTS[net]) == 0:
			INDEX_LISTS[net] = [i for i in range(len(labels)) if (labels[i] in CLASS_LISTS[net] or np.random.uniform() < LEAK_CHANCE)]
			#INDEX_LIST[0] = [i for i in range(len(labels))]
		inds.append(INDEX_LISTS[net].pop(random.randint(0,len(INDEX_LISTS[net])-1)))
	
	truths = labels[inds]
	return(data[inds,:,:,:],truths)
	
	
def eval(nets,eval_data,eval_labels,inputs,truths,is_train,sess):
	acc_counts = [[] for i in range(len(nets))]
	accs = [[] for i in range(len(nets))]
	acc_final = []
	
	for item in range(len(eval_labels)):
		feed_dict = {is_train:False}		
		
		for net in range(len(nets)):
			feed_dict[inputs[net]] = eval_data[np.newaxis,item,:,:,:]
			feed_dict[truths[net]] = eval_labels[np.newaxis,item]

		ans = sess.run(nets,feed_dict=feed_dict)
		for net in range(len(nets)):
			acc_counts[net].append((eval_labels[item] not in CLASS_LISTS[net]) if (np.argmax(ans[net])==len(CLASS_LISTS[net])) else CLASS_LISTS[net][np.argmax(ans[net])]==eval_labels[item])
		final_ans = np.concatenate((ans[0][:,:-1],ans[1][:,:-1]),axis=1)
		#print(final_ans)
		sorting = np.concatenate((CLASS_LISTS[0],CLASS_LISTS[1]),axis=0)
		sorting = np.argsort(sorting)
		final_ans = final_ans[:,sorting]
		acc_final.append(np.argmax(final_ans) == eval_labels[item])
					
	for net in range(len(nets)):
		accs[net] = np.mean(acc_counts[net])
		print('Net %d: Acc %.4f' %(net,accs[net]))
	print('Final System: Acc %.4f' % np.mean(acc_final))
	
	file_print(','.join(["%.5f"%accs[net] for net in range(len(nets))] + ['%.5f' % np.mean(acc_final)]),'self_assign_err.csv')
	
	
def net_architecture(inputs,is_train,net):	
	in_chann = 1
	feat = inputs                                         
	for block in range(2):
		for lay in range(3):
			out_chann = CONV_WIDTHS[net]*(block+1)
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
	
	fc_weights = tf.get_variable('fc_weights',[in_chann,FC_WIDTHS[net]],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)))
	fc_bias = tf.get_variable('fc_bias',[FC_WIDTHS[net]],tf.float32,initializer=tf.constant_initializer(0.01))
	feat = tf.matmul(feat,fc_weights)
	feat = tf.nn.bias_add(feat,fc_bias)
	feat = tf.nn.relu(feat)
	feat = tf.layers.batch_normalization(feat,training=is_train,renorm=True)
	feat = tf.nn.dropout(feat,1-DROP_PROB)
	
	return feat
