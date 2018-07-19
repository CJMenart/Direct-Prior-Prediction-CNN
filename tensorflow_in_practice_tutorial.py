"tutorial from 7/19/18 on TF. Examples sort of go in order. Many also have common mistakes inserted with the correction commented out."
#remember--the documentation is your first resource! 

import tensorflow as tf
import numpy as np
import random
import os

def hello_world():
	"In tensorflow, you define variables, and operations called nodes, in a structure called a graph. These are not evaluated when you construct them."
	hello = tf.constant('Hello World!')
	sess = tf.InteractiveSession()
	print(sess.run(hello))
	
	#Or
	sess.close()
	with tf.Session() as sess:
		print(sess.run(hello))
		

def hello_math():
	"Typical tensorflow variable/node construction might look like this. Don't forget to initialize the graph--but be careful of initializing after loading a saved graph!"
	sess = tf.InteractiveSession()
	
	data = tf.constant([[-1,1,5,-5,20],[0,10,-10,-5,2]],dtype=tf.float32)
	#you should use get_variable to make variables. Not tf.Variable.
	num_in = 5
	num_out = 1
	weights = tf.get_variable('weights',[num_in,num_out],initializer=tf.truncated_normal_initializer(0,0.2)) #default dtype for initailizer is float32
	output = tf.matmul(data,weights)
	
	#don't forget to initialize!
	#tf.global_variables_initializer().run()
	print(sess.run(output))
	sess.close()


def hello_overloading():
	"Tensorflow overloads some common operators, like PEMDAS and slicing, into TF node constructors."
	sess = tf.InteractiveSession()
	
	data = tf.constant([[-1,1,5,-5,20],[0,10,-10,-5,2]],dtype=tf.float32)
	data_squared = tf.multiply(data,data)
	data_squared_again = data*data
		
	#tf.global_variables_initializer().run()  #note this is not needed here
	print(sess.run(data_squared))
	print(sess.run(data_squared_again))
	sess.close()
	
	
def cannot_np_in_graph():
	"It's easy to forget that you can ONLY use tensorflow node constructors to manipulate your variables when constructing a computation graph."
	sess = tf.InteractiveSession()
	
	data = tf.constant([[-1,1,5,-5,20],[0,10,-10,-5,2]],dtype=tf.float32)
	num_in = 5
	num_out = 1
	
	#this will fail b/c indexing python arrays with other arrays is not overloaded in tensorflow operators
	weights = tf.get_variable('weights',[num_in,num_out*2],initializer=tf.truncated_normal_initializer(0,0.2)) 
	index_into_weights = [0,2,5,6,7]
	output = tf.matmul(data,weights[index_into_weights])
	
	tf.global_variables_initializer().run()
	print(sess.run(output))
	sess.close()
	
	
def hello_optimization():
	"Typically you will train your variables using an optimizer. It might look like this. This script also shows a plethora of errors--I made three just writing this simple example!"
	sess = tf.InteractiveSession()
	
	data = tf.constant([[-1,1,5,-5,20],[0,10,-10,-5,2]],dtype=tf.float32)
	#you should use get_variable to make variables. Not tf.Variable.
	num_in = 5
	num_out = 1
	weights = tf.get_variable('weights',[num_in,num_out],initializer=tf.truncated_normal_initializer(0,0.2)) #default dtype for initailizer is float32
	output = tf.matmul(data,weights)
	
	#tf.pow broadcasts the scalar power, '2', to all elements of out array
	loss = tf.pow(output - tf.constant([2,-2],dtype=tf.float32),2)
	#don't forget to use reduce_sum to that loss is a scalar!
	#loss = tf.reduce_sum(tf.pow(output - tf.constant([2,-2],dtype=tf.float32),2))
	#watch out for automatic broadcasting!
	#loss = tf.reduce_sum(tf.pow(output - tf.constant([[2],[-2]],dtype=tf.float32),2))
	optimizer = tf.train.GradientDescentOptimizer(1e-2)
	#watch out for model divergence with high learning rates
	#optimizer = tf.train.GradientDescentOptimizer(1e-4)
	train = optimizer.minimize(loss)
	#OR the below 2 ops are equivalent to 'train'--but allow you to insepct/manipulate gradients
	gradients = optimizer.compute_gradients(loss)
	apply_grad = optimizer.apply_gradients(gradients)
	
	tf.global_variables_initializer().run()
	for iter in range(1000):
		sess.run(train)
		print(sess.run(output))
		
		#when debugging, sess.run allows print-style debugging. Inspect things!
		#print(sess.run(loss))
		#print(sess.run(weights))
	sess.close()
	
	
def hello_batchnorm():
	"Batch normalization is a common and powerful technique in deep learning today (~2018). Here's how you can do it in Tensorflow. This also introduces feed_dicts."
	
	sess = tf.InteractiveSession()
	
	data = tf.constant([[-1,1,5,-5,20],[0,10,-10,-5,2]],dtype=tf.float32)
	is_train = tf.placeholder(tf.bool,None)
	num_in = 5
	num_out = 1
	weights = tf.get_variable('weights',[num_in,num_out],initializer=tf.truncated_normal_initializer(0,0.2)) #default dtype for initailizer is float32
	output = tf.matmul(data,weights)
	normed_output = tf.layers.batch_normalization(output,training=is_train) #there are interesting optional parameters for this function
	#Also, some people take an alternate approach to specifying train/test mode by creating two separate instances of their network which share variables, instead of using placeholder values for is_train. Those people are stupid.

	loss = tf.reduce_sum(tf.pow(normed_output - tf.constant([[2],[-2]],dtype=tf.float32),2))
	optimizer = tf.train.GradientDescentOptimizer(1e-4)
	train = optimizer.minimize(loss)	
	
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	train = tf.group(train, *update_ops)

	#but moral of this story: don't batch norm your final layer!
	tf.global_variables_initializer().run()
	for iter in range(1000):
		sess.run(train,feed_dict={is_train:True})
		print(sess.run(normed_output,feed_dict={is_train:True}))
		
	sess.close()
	

def think_about_gradients():
	"Tensorflow automatically computes the gradients of your graph, a very powerful and useful feature. If you don't understand how gradients work you will use it to powerfully make your life suck. This example also shows typical elements from a classification network, like softmax output."
	
	sess = tf.InteractiveSession()
	
	#simple linear classification here
	data = tf.constant([[0,1,3,-5,1],
						[0,1,-1,-3,2],
						[1,1,1,1,-2],
						[0,-3,-3,4,-3]],dtype=tf.float32)
	targets = tf.constant([[0,0,1],
						  [0,0,1],
						  [0,1,0],
						  [1,0,0]],dtype=tf.float32)
	weights = tf.get_variable('weights',[5,3],initializer=tf.truncated_normal_initializer(0,10)) #weights are significantly too large
	activation = tf.matmul(data,weights)
	output = tf.nn.softmax(activation,-1) #axis=-1 means 'apply softmax along the last (class) dimension, not the batch dimension.'
	
	cross_entropy_loss = tf.reduce_sum(-1.0*targets*tf.log(output))
	#ok, we need an epsilon value to prevent infinite loss
	#cross_entropy_loss = tf.reduce_sum(-1.0*targets*tf.log(tf.maximum(output,1e-8)))
	#If the output goes through the constant branch of tf.maximum, there is no gradient w.r.t variables!
	#cross_entropy_loss = tf.reduce_sum(-1.0*targets*tf.log(tf.minimum(1.0,output+1e-8)))
	#remember--training models using gradient descent is finicky! Even the dead-simple linear model above cannot be learnt if the initial weights are in an inappropriate range! We need to reduce the standard deviation of the weights to learn successfully. For if the output is totally saturated into a bad regime, there may be no gradients, and no saving the model. But to a great degree batch norm helps with exactly this problem.
	
	optimizer = tf.train.GradientDescentOptimizer(1e-1)
	gradients = optimizer.compute_gradients(cross_entropy_loss)
	apply_grad = optimizer.apply_gradients(gradients)
	
	tf.global_variables_initializer().run()
	for iter in range(10000):
		sess.run(apply_grad)
		print(sess.run(output))
		#print(sess.run(gradients))
		print(sess.run(cross_entropy_loss))
		
	sess.close()	
	

def hello_mnist():
	"Example of a complete neural network, with convolutional layers and all the bells and whistles."

	#sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
	
	with tf.Session() as sess:
	
		#load data
		mnist = tf.contrib.learn.datasets.load_dataset("mnist")
		train_data = mnist.train.images # Returns np.array
		train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
		eval_data = mnist.test.images # Returns np.array
		eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
		train_data = np.reshape(train_data,[-1,28,28,1])
		eval_data = np.reshape(eval_data,[-1,28,28,1])
		#shrink test set for runtime. Every 4th item.
		eval_data = eval_data[::4,:]
		eval_labels = eval_labels[::4]
		
		#all the placeholders we will be using.
		is_train = tf.placeholder(tf.bool,None)
		inputs = tf.placeholder(tf.float32,[None,28,28,1])
		truths = tf.placeholder(tf.int64,[None])
		
		#network architecture
		in_chann = 1
		WIDTH = 32
		feat = inputs                                         
		for block in range(2):
			for lay in range(3):
				out_chann = WIDTH*(block+1)
				conv_weights = tf.get_variable('conv%d_%d_weights' % (block,lay),[3,3,in_chann,out_chann],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/(9*in_chann))),regularizer = tf.contrib.layers.l2_regularizer(1e-4))
				_activation_summary(conv_weights)
				bias = tf.get_variable('conv%d_%d_bias' % (block,lay),[out_chann],tf.float32,initializer=tf.constant_initializer(0.01))
				feat = tf.nn.conv2d(feat,conv_weights,[1,1,1,1],'SAME')
				feat = tf.nn.bias_add(feat,bias)
				feat = tf.nn.relu(feat)
				feat = tf.layers.batch_normalization(feat,training=is_train,renorm=True)
				in_chann = out_chann
					
			feat = tf.nn.max_pool(feat,[1,2,2,1],[1,2,2,1],'SAME')
		
		#feat = tf.reduce_mean(feat,axis=[1,2])  #another option for going from convolution to fully-connected
		feat = tf.contrib.layers.flatten(feat)

		FC_WIDTH = 256
		DROP_PROB = 0.1
		in_chann = feat.shape.as_list()[-1]	
		fc_weights = tf.get_variable('fc_weights',[in_chann,FC_WIDTH],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/in_chann)),regularizer = tf.contrib.layers.l2_regularizer(1e-4))
		_activation_summary(fc_weights)
		fc_bias = tf.get_variable('fc_bias',[FC_WIDTH],tf.float32,initializer=tf.constant_initializer(0.01))
		feat = tf.matmul(feat,fc_weights)
		feat = tf.nn.bias_add(feat,fc_bias)
		feat = tf.nn.relu(feat)
		feat = tf.layers.batch_normalization(feat,training=is_train,renorm=True)
		feat = tf.nn.dropout(feat,1-DROP_PROB)
		
		in_chann = FC_WIDTH
		NCLASS = 10
		final_weights = tf.get_variable('final_weights',[in_chann,NCLASS],tf.float32,initializer=tf.truncated_normal_initializer(np.sqrt(2/FC_WIDTH)),regularizer = tf.contrib.layers.l2_regularizer(1e-4))
		_activation_summary(final_weights)
		final_bias = tf.get_variable('final_bias',[NCLASS],tf.float32,initializer=tf.constant_initializer(0.01))
		feat = tf.matmul(feat,final_weights)
		feat = tf.nn.bias_add(feat,final_bias)
		net = tf.nn.softmax(feat,-1)
		_activation_summary(net,'output')
		
		expanded_truths = tf.one_hot(truths,NCLASS)
		cross_entropy_loss = tf.reduce_sum(-1.0*expanded_truths*tf.log(tf.minimum(1.0,net+1e-8)))
		total_loss = cross_entropy_loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
				
		train = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train = tf.group(train,*update_ops)
		tf.global_variables_initializer().run()

		summaries = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter('',graph=sess.graph)
		
		#slim, keras, and probably 4 other ways to do things also allow this loop to be encapsulated. I like to be at this level of control.
		indices = []
		for iter in range(40001):
			feed_dict = {is_train:True}
			feed_dict[inputs],feed_dict[truths] = _get_batch(train_data,train_labels,64,indices)
			_,smry = sess.run([train,summaries],feed_dict=feed_dict)
			summary_writer.add_summary(smry, iter)
			if iter % 250 == 0:
				_eval(net,eval_data,eval_labels,inputs,truths,is_train,sess)
	
def _get_batch(data,labels,batch_size,indices):
	"Helper function for hello_mnist. Return batches, iterating through epoch in random order using numpy."
	
	inds = []
	for i in range(batch_size):
		if len(indices) == 0:
			indices = [i for i in range(len(labels))]
		inds.append(indices.pop(random.randint(0,len(indices)-1)))
	
	return(data[inds,:,:,:],labels[inds])
	
def _eval(net,eval_data,eval_labels,inputs,truths,is_train,sess):
	"helper function for hello_mnist. Evaluate accuracy on mnist test data."
	acc_counts = []
	
	for item in range(len(eval_labels)):
		feed_dict = {is_train:False}		
		feed_dict[inputs] = eval_data[np.newaxis,item,:,:,:]
		feed_dict[truths] = eval_labels[np.newaxis,item]

		ans = sess.run(net,feed_dict=feed_dict)
		acc_counts.append(np.argmax(ans) == eval_labels[item])
					
	print('Acc %.4f' % np.mean(acc_counts))
	
def _activation_summary(x,name=None):
    "\
    :param x: A Tensor\
	:param name: optional name that will show up in Tensorboard. Recommended if you have many summaries.\
    :return: Add histogram summary and scalar summary of the sparsity of the tensor\
    "
    if name is None:
        name = x.op.name
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.scalar(name + '/max', tf.reduce_max(x))
    tf.summary.scalar(name + '/min', tf.reduce_min(x))

#at this point in the talk, to go build_tf_record to talk about tfrecords
#talk about saving and restoring?

def control_gpu_usage():
	"Particularly useful for those who own a lab machine with multiple GPU cards. Normally, Tensorflow uses ALL gpus when running. To run multiple instances of TF at the same time, restrict each one to a specific GPU using this call."

	gpu = 0
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
	
	sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
	hello = tf.constant([1,2],name='hello')	
	print(hello)
	print(sess.run(hello))
	
	
def variable_reuse():
	"Variable re-use is one of the reasons you should use get_variable. It can be used for a variety of purposes, such as creating siamese networks."
	
	sess = tf.InteractiveSession()
	
	input1 = tf.placeholder(tf.float32,[None,10])
	input2 = tf.placeholder(tf.float32,[None,10])
	
	#create two totally separate models
	with tf.variable_scope("model1") as scope:
		model1 = _linear_model(input1)
	with tf.variable_scope("model2") as scope:
		model2 = _linear_model(input2)
		
	#create two models with different inputs that share weights
	with tf.variable_scope("model") as scope:
		model1 = _linear_model(input1)
		scope.reuse_variables()
		model2 = _linear_model(input2)

	#this will throw an exception because you tried to create a variable that already exists
	with tf.variable_scope("wrong") as scope:
		model1 = _linear_model(input1)
		model2 = _linear_model(input2)
		
def _linear_model(input):
	"Helper for variable_reuse. declared a linear model with matrix multiplication."
	in_chann = input.shape.as_list()[-1]
	weights = tf.get_variable('weights',[in_chann,1],initializer=tf.truncated_normal_initializer(0,0.2))
	return tf.matmul(input,weights)
	
def save_and_restore():
	"Periodically saving and restoring models is an important part of training larger systems."
	sess = tf.InteractiveSession()
	
	weights = tf.get_variable('weights',[10,1],initializer=tf.truncated_normal_initializer(0,0.2))
	
	latest_model = _get_latest_model('')
	if latest_model:
		tf.global_variables_initializer().run()
		saver.restore(sess,latest_model)
		start = int(latest_model.split('-')[-1])+1
	else:
		double_print('Creating a new network...',text_log)
		tf.global_variables_initializer().run()	#network.load_weights(os.path.join(data_loader.base_fcn_weight_dir(),net_opts['fcn_weight_file']),sess)
	
	saver.save(sess, model_name,global_step = 0)
	
def _get_latest_model(checkpoint_dir):
	"If you move away from the 'checkpoint' file, and just have a .ckpt, this can try to find it. Helper for save_and_restore."
	latest_model = tf.train.latest_checkpoint(checkpoint_dir)
	if not latest_model: #attempt to find model file directly
		filenames = os.listdir(checkpoint_dir)
		matches = fnmatch.filter(filenames,"*.meta*")
		if len(matches) > 0:
			latest_model = matches[-1]
		if latest_model:
			latest_model = os.path.join(checkpoint_dir,latest_model[:-5])
	return latest_model