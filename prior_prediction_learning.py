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
#turns off annoying warnings about compiling TF for vector instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

DEBUG = True
# Chris Menart, 1-9-18
#went back to editing 5-18-18		

#TODO write testing control method IN PROGRESS
#testing script for these purposes should write out test predictions, but also spread for all values 
#(runs one net--to evaluate spread for multiple clusters we will call this multiple times)
def testing(net_opts,checkpoint_dir):
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
		data_loader = CVL2018DataLoader(net_opts['base_fcn_weight_dir'],net_opts['dataset_dir'])	
	elif net_opts['data_loader_type'] == 'TFRecord':
		data_loader = CVL2018TFRecordDataLoader(net_opts['base_fcn_weight_dir'],net_opts['dataset_dir'],
			1 if net_opts['img_sizing_method'] == 'run_img_by_img' else net_opts['batch_size'])
	else:
		raise Exception("data_loader type not recognized.")
	
	is_train = tf.placeholder(tf.bool,None)
	network = net.PriorNet(net_opts,data_loader.num_labels(),data_loader.inputs(),data_loader.seg_target(),is_train)
	
	best_val_loss = tf.Variable(sys.float_info.max,trainable=False,name="best_val_loss")
	
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
	num_test = data_loader.num_data_items(partition_enum.TEST)
	for t_ind in range(num_test):
					
		feed_dict = data_loader.feed_dict(partition_enum.TEST,batch_size=1)  
		feed_dict[is_train] = False

		prior = sess.run(network.prior, feed_dict=feed_dict)
		priors.append(prior)

	fname = os.path.join(checkpointDir,"test_priors.csv")
	file = open(fname,"w+")
	for line in embed:
		print(','.join(["%.6f"%n for n in line]),file=file)
	file.close()		
	sess.close()
	double_print("Done.",text_log)



#TODO OUT OF DATE
def train_on_clusters(paths,net_opts):
	"Trains a network on the entire dataset, then recurses into a sub-directory and fine-tunes a copy on each specified cluster within that dataset."
	
	#TODO: you will need a new graph for every train, which will require constructing a new data_loader
	
	#global train
	train_img_names = csvr.readall(paths['train_name_file'],csvr.READ_STR)
	val_img_names = csvr.readall(paths['train_name_file'],csvr.READ_STR)
	training(paths,net_opts,train_img_names,val_img_names)
	
	train_clusters = csvr.readall(paths['train_clusters_file'],csvr.READ_INT)
	val_clusters = csvr.readall(paths['val_clusters_file'],csvr.READ_INT)
	num_clust = np.max(train_clusters)
	for clust in range(clusters):
	
		clust_opts = dict(net_opts)
		clust_paths = dict(paths)
		clust_dir = os.path.join(checkpoint_dir,'clust_%d' % clust)
		clust_checkpoint_dir = clust_dir
		if not os.path.exists(clust_dir):
			copyanything(global_dir,clust_dir)
		clust_opts['max_iter'] = net_opts['max_iter']*2
		
		clust_train_img_names = [train_img_names[n] for n in len(train_img_names) if train_clusters[n] == clust]
		clust_val_img_names = [val_img_names[n] for n in len(val_img_names) if val_clusters[n] == clust]
		
		training(clust_paths,clust_opts,clust_train_img_names,clust_val_img_names)
	
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
		data_loader = CVL2018DataLoader(net_opts)	
	elif net_opts['data_loader_type'] == 'TFRecord':
		data_loader = CVL2018TFRecordDataLoader(net_opts)
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
		
	trainable_vars = tf.trainable_variables()	
	if DEBUG:
		print("Trainable Variables:")
		print(trainable_vars)
			
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
				feed_dict = data_loader.feed_dict(partition_enum.VAL,batch_size=batch_size)  
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
