"Controls training/testing loop for deep learning of priors."

#import time
import numpy as np
import tensorflow as tf
import os
import random
import sys
import prior_net as net
import numpy as np
from copyanything import *
from print_to_file import *
from class_frequency import *
from train_ops import *
from get_latest_model import *
from data_loader import *
import partition_enum
from build_feed_dict import *
from augment_img_node import *
from cvl_2018_data_loader import *
from cvl_2018_tfrecord_data_loader import *
import partition_enum
#turns off annoying warnings about compiling TF for vector instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

DEBUG = False
# Chris Menart, 1-9-18
#went back to editing 5-18-18		


#testing script for these purposes should write out test predictions, but also spread for all values 
#(runs one net--to evaluate spread for multiple clusters we will call this multiple times)
def evaluate(net_opts,checkpoint_dir,partition):
	"Run network, producing predictions for all test values (and spread of evaluations w/dropout if applicable)."
	
	#Paths to text output logs
	text_log = os.path.join(checkpoint_dir, "NetworkLog.txt")
	double_print("Welcome to prior network testing.",text_log)	
		
	#first, start TF session and construct data_loader so we can begin setting up data
	if net_opts['is_gpu']:
		sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
	else:
		#debug to force CPU mode!
		config = tf.ConfigProto(
			device_count = {'GPU': 0}
		)
		sess = tf.InteractiveSession(config=config)
		
	if net_opts['data_loader_type'] == 'CVL_2018':
		data_loader = CVL2018DataLoader(net_opts,False)	
	elif net_opts['data_loader_type'] == 'TFRecord':
		data_loader = CVL2018TFRecordDataLoader(net_opts,False)
	else:
		raise Exception("data_loader type not recognized.")
	
	is_train = tf.placeholder(tf.bool,None)	
	repeated_inputs = tf.tile(data_loader.inputs(),[net_opts['num_dropout_eval_reps'],1,1,1])
	repeated_truths = tf.tile(data_loader.seg_target(),[net_opts['num_dropout_eval_reps'],1,1])
	spread_network = net.PriorNet(net_opts,data_loader.num_labels(),repeated_inputs,repeated_truths,is_train)
	_,spread = tf.nn.moments(spread_network.prior,axes=[0])
	
	with tf.variable_scope(tf.get_variable_scope(), reuse=True):
		net_opts['drop_prob'] = 0
		final_network = net.PriorNet(net_opts,data_loader.num_labels(),data_loader.inputs(),data_loader.seg_target(),is_train)
	
	best_val_loss = tf.Variable(sys.float_info.max,trainable=False,name="best_val_loss")
	
	if True:
		trainable_vars = tf.trainable_variables()	
		print("Trainable Variables:")
		print(trainable_vars)
		
	saver = tf.train.Saver()	
	latest_model = get_latest_model(checkpoint_dir)
	if latest_model:
		tf.global_variables_initializer().run()
		saver.restore(sess,latest_model)
		start = int(latest_model.split('-')[-1])+1
		double_print('Starting from iteration %d. Current validation loss: %.5f' % (start, np.mean(sess.run([best_val_loss]))),text_log)
	else:
		raise Exception('Trained model not found.')
		
	priors = []
	spreads = []
	num_test = data_loader.num_data_items(partition)
	for t_ind in range(num_test):
					
		feed_dict = data_loader.feed_dict(partition,batch_size=1)  
		feed_dict[is_train] = False

		if net_opts['is_eval_spread']:
			prior_ev, spread_ev = sess.run([final_network.prior, spread], feed_dict=feed_dict)
			print(np.squeeze(spread_ev))
			spreads.append(np.squeeze(spread_ev))
		else:
			prior_ev = sess.run(final_network.prior, feed_dict=feed_dict)		
		priors.append(np.squeeze(prior_ev))
		
		double_print('Image %d' % t_ind,text_log)
		#double_print(priors,text_log)
		
	fname = os.path.join(checkpoint_dir,"%s_priors.csv" % partition_enum.SPLITNAMES[partition])
	file = open(fname,"w+")
	for line in priors:
		print(','.join(["%.6f"%n for n in line]),file=file)
	file.close()
	if net_opts['is_eval_spread']:
		fname = os.path.join(checkpoint_dir,"%s_spreads.csv" % partition_enum.SPLITNAMES[partition])
		file = open(fname,"w+")
		for line in spreads:
			print(','.join(["%.6f"%n for n in line]),file=file)
		file.close()	
		
	sess.close()
	double_print("Done.",text_log)
	
	
def training(net_opts,checkpoint_dir):
	"Train the actual network here. Can restart training from checkpoint if stopped.\
	Gnarliest function in the software probably."
	
	#Paths to text output logs
	text_log = os.path.join(checkpoint_dir, "NetworkLog.txt")
	train_err_log = os.path.join(checkpoint_dir, "TrainErr.csv")
	val_err_log = os.path.join(checkpoint_dir, "val_err.csv")
	double_print("Welcome to prior network training.",text_log)	
		
	#first, start TF session and construct data_loader so we can begin setting up data
	if net_opts['is_gpu']:
		sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
	else:
		#debug to force CPU mode!
		config = tf.ConfigProto(
			device_count = {'GPU': 0}
		)
		sess = tf.InteractiveSession(config=config)
		
	if net_opts['data_loader_type'] == 'CVL_2018':
		data_loader = CVL2018DataLoader(net_opts,True)	
	elif net_opts['data_loader_type'] == 'TFRecord':
		data_loader = CVL2018TFRecordDataLoader(net_opts,True)
	else:
		raise Exception("data_loader type not recognized.")
	
	if net_opts['remapping_loss_weight'] > 0:
		map_mat = data_loader.map_mat()
	#more debugging
	if DEBUG:
		double_print('Example validation item',text_log)
		ex_im,ex_truth = data_loader.img_and_truth(0,partition_enum.VAL)
		double_print(ex_im,text_log)
		double_print(ex_truth,text_log)
	
	#compute average of classes present for binary loss weights
	if not net_opts['is_target_distribution'] and net_opts['is_loss_weighted_by_class']:
		double_print('Computing class frequencies...',text_log)
		class_freq = class_frequency(data_loader)
		double_print(class_freq,text_log)
	else:
		#TODO move this to selecting weighted or unweighted loss function
		class_freq = np.ones((1,data_loader.num_labels()),dtype=np.float32)/2 #unweight

	#Augmentation, sizing--but not mean-subtraction b/c it is specific to network architecture. Fingers crossed that doesn't cause issues later
	inputs,seg_target = augment_no_size_change(data_loader.inputs(),data_loader.seg_target())
	is_train = tf.placeholder(tf.bool,None)
	
	network = net.PriorNet(net_opts,data_loader.num_labels(),inputs,seg_target,is_train,class_frequency=class_freq)
	
	best_val_loss = tf.Variable(sys.float_info.max,trainable=False,name="best_val_loss")
	reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	total_loss = network.loss + reg_loss
	if DEBUG:
		double_print('Regularization Losses:',text_log)
		double_print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),text_log)
	
	train_op_handler = TrainOpHandler(net_opts,total_loss)
	#tensorboard inspection
	summaries = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(checkpoint_dir,graph=sess.graph)	
	saver = tf.train.Saver()
	
	if DEBUG:
		trainable_vars = tf.trainable_variables()	
		print("Trainable Variables:")
		print(trainable_vars)

		
	latest_model = get_latest_model(checkpoint_dir)
	if latest_model:
		tf.global_variables_initializer().run()
		saver.restore(sess,latest_model)
		start = int(latest_model.split('-')[-1])+1
		double_print('Starting from iteration %d. Current validation loss: %.5f' % (start, np.mean(sess.run([best_val_loss]))),text_log)
	else:
		double_print('Creating a new network...',text_log)
		start = 0
		tf.global_variables_initializer().run()
		network.load_weights(os.path.join(data_loader.base_fcn_weight_dir(),net_opts['fcn_weight_file']),sess)
	model_name = os.path.join(checkpoint_dir,net_opts['model_name'])
					
	for iter in range(start,net_opts['max_iter']):
		batch_size = net_opts['batch_size']
		if iter == net_opts['iter_end_only_training']:
			double_print('Switching to end-to-end training.',text_log)
				
		if iter % (net_opts['batches_per_val_check']) == 0:
			#TODO: Encapsulate out all statistics you want somehow, so that changing the list doesn't change code in a dozen places
			val_loss = 0
			val_err = 0
			val_acc = 0
			val_num = 0
			step_sz = 1 if net_opts['img_sizing_method']=='run_img_by_img' else batch_size
			num_val = data_loader.num_data_items(partition_enum.VAL)
			if DEBUG:
				print('num_val')
				print(num_val)
			for v_ind in range(0,num_val,step_sz):
							
				# BIG WARNING: Right now, validation set we get is 'approximate' in the sense that if the batch size doesn't divide
				# evenly into val set size, we'll be missing a few validation images from each set. Not necessarily the same ones each time.
				# if we have a large val set, this should not have any noticeable impact on training, excpet when we're using val stats to make
				# sure the network is actually changing. This is a casualty of making the code work nicely with both TFRecords and feed_dicts.
				feed_dict = data_loader.feed_dict(partition_enum.VAL,batch_size=step_sz)  
				feed_dict[is_train] = False        
				#feed_dict = build_feed_dict(data_loader,network,range(v_ind,min(num_val,v_ind+step_sz)),partition_enum.VAL,net_opts)

				if DEBUG:
					print('val feed_dict:')
					print(feed_dict)
				loss, err, acc, smry = sess.run([network.loss, network.prior_err, network.seg_acc, summaries], feed_dict=feed_dict)
				val_loss += loss
				val_err += err
				val_acc += acc
				val_num += step_sz
				
			val_loss = np.mean(val_loss)/val_num
			val_err = np.mean(val_err)/val_num
			val_acc = np.mean(val_acc)/val_num
			double_print('step %d: val loss %.5f' % (iter, val_loss),text_log)
			double_print('step %d: val error ~= %.5f' % (iter, val_err),text_log)
			double_print('step %d: val acc ~= %.5f' % (iter,val_acc),text_log)
			file_print(val_err,val_err_log)
			summary_writer.add_summary(smry, iter)
			new_best = False
			if val_loss < np.mean(sess.run([best_val_loss])):
				sess.run([best_val_loss.assign(val_loss)])
				double_print("NEW VALIDATION BEST",text_log)
				new_best = True
			if new_best or iter % net_opts['iter_per_automatic_backup'] == 0:
				saver.save(sess, model_name,global_step = iter)
		
		#REST OF LOOP: TRAINING STEP
		
		if net_opts['img_sizing_method']=='run_img_by_img':
			loss = 0
			err = 0
			acc = 0
			for t_ind in range(batch_size):
			
				#feed_dict = build_feed_dict(data_loader,network,[t_ind],partition_enum.TRAIN,net_opts)
				feed_dict = data_loader.feed_dict(partition_enum.TRAIN,batch_size=1)
				feed_dict[is_train] = True
				#feed_dict = build_feed_dict(data_loader,network,[t_ind],partition_enum.TRAIN,net_opts)
				
				#actual train step
				_, cur_loss, cur_err,cur_acc = sess.run([train_op_handler.train_op(iter),network.loss,network.prior_err,network.seg_acc], feed_dict=feed_dict)
				loss += cur_loss
				err += cur_err
				acc += cur_acc
		else:
			feed_dict = data_loader.feed_dict(partition_enum.TRAIN,batch_size=batch_size)        
			feed_dict[is_train] = True
			#feed_dict = build_feed_dict(data_loader,network,t_inds,partition_enum.TRAIN,net_opts)
			#actual train step
			_, loss, err,acc = sess.run([train_op_handler.train_op(iter),network.loss,network.prior_err,network.seg_acc], feed_dict=feed_dict)     		
			
		train_op_handler.check_gradients(iter,sess)
		train_op_handler.post_batch_actions(iter,sess)
		
		loss = loss/batch_size #avg loss per element, to print
		err = err/batch_size
		acc = acc/batch_size
		double_print('step %d: loss %.3f, p-err %.3f, acc = %.3f' % (iter, loss,err,acc),text_log)
		file_print(err,train_err_log)
		
		if np.isnan(loss) or np.isnan(err):
			# NOTE: It is theoretically possible,
			# though unlikely, to have a model diverge with NaN loss without a bug
			# if the weights cause an output of the network to overflow to inf
			# plus y'know we want to catch bugs
			double_print('Model diverged with loss = %.2f and err = %.2f' % (loss,err),text_log)	
			quit()			
		
	sess.close()
	double_print("Done.",text_log)
	

def train_on_clusters(net_opts,checkpoint_dir):
	"Recurses into a sub-directory OF AN ALREADY-TRAINED NETWORK and fine-tunes a copy on each specified cluster within that dataset. Be careful to specifiy a limited number of training iterations for this one!"
	for clust in range(net_opts['num_clusters']):
		tf.reset_default_graph()
		clust_net_opts = net_opts.copy()
		clust_net_opts['cluster'] = clust
		clust_net_opts['max_iter'] = net_opts['max_iter']*2
		clust_dir = os.path.join(checkpoint_dir,'clust_%d' % clust)
		if not os.path.exists(clust_dir):
			copyanything(checkpoint_dir,clust_dir)
		training(clust_net_opts, clust_dir)

		
def evaluate_on_clusters(net_opts,checkpoint_dir,partition):
	"Recurses into the trained sub-directories of a network, which were fine-tuned on clusters using train_on_clusters, and evaluates each."
	for clust in range(net_opts['num_clusters']):
		tf.reset_default_graph()
		clust_net_opts = net_opts.copy()
		clust_net_opts['cluster'] = clust
		clust_dir = os.path.join(checkpoint_dir,'clust_%d' % clust)
		assert(os.path.exists(clust_dir))
		evaluate(clust_net_opts, clust_dir, partition)

