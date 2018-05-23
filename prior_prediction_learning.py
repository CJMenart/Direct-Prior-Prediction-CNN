"Controls training/testing loop for deep learning of priors."

#import time
import numpy as np
import tensorflow as tf
import os
import random
import sys
import cv2
import csvreadall as csvr
import prior_net as net
from augment_img import *
import numpy as np
from copyanything import *
from print_to_file import *
from class_frequency import *
from train_ops import *
from my_load_mat import *
from get_latest_model import *
#turns off annoying warnings about compiling TF for vector instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

DEBUG = True
# Chris Menart, 1-9-18
#went back to editing 5-18-18		

#TODO write testing control method


#TODO not tested since update
def train_on_clusters(paths,net_opts):
	"Trains a network on the entire dataset, then recurses into a sub-directory and fine-tunes a copy on each specified cluster within that dataset."
	
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
		clust_dir = os.path.join(paths['checkpoint_dir'],'clust_%d' % clust)
		clust_paths['checkpoint_dir'] = clust_dir
		if not os.path.exists(clust_dir):
			copyanything(global_dir,clust_dir)
		clust_opts['max_iter'] = net_opts['max_iter']*2
		
		clust_train_img_names = [train_img_names[n] for n in len(train_img_names) if train_clusters[n] == clust]
		clust_val_img_names = [val_img_names[n] for n in len(val_img_names) if val_clusters[n] == clust]
		
		training(clust_paths,clust_opts,clust_train_img_names,clust_val_img_names)

		
def train_on_all(paths,net_opts):
	"Wrapper for 'training' to just train once on whole dataset."
	#TODO: Pull out reading/filetype-specific logic into own function
	train_img_names = csvr.readall(paths['train_name_file'],csvr.READ_STR)
	val_img_names = csvr.readall(paths['train_name_file'],csvr.READ_STR)

	training(paths,net_opts,train_img_names,val_img_names)	
	
	
def training(paths,net_opts,train_img_names,val_img_names):
	"Train the actual network here. Can restart training from checkpoint if stopped."
	
	#Settings and paths and stuff			
	text_log = os.path.join(paths['checkpoint_dir'], "NetworkLog.txt")
	train_err_log = os.path.join(paths['checkpoint_dir'], "TrainErr.csv")
	val_err_log = os.path.join(paths['checkpoint_dir'], "val_err.csv")
	double_print("Welcome to prior network training.",text_log)	
	if net_opts['remapping_loss_weight'] > 0:
		map_mat = np.array(csvr.readall(paths['map_mat_file']),csvr.READ_FLOAT)
	
	double_print("Loading validation images...",text_log)

	#TODO: When you factor out image-getting, factor out pre-loading as well...
	val_imgs = []
	val_targets = []
	val_remap_targets = []
	val_base_probs = []
	for v in range(len(val_img_names)):
		img = cv2.imread(os.path.join(paths['im_dir'], val_img_names[v]))
		sz = img.shape
		val_imgs.append(img[np.newaxis,:,:,:])
		
		truth = my_load_int_mat(os.path.join(paths['truth_dir'], val_img_names[v][:-3] + 'mat'))		
		val_targets.append(truth[np.newaxis,:,:])
		
		if net_opts['remapping_loss_weight'] > 0:
			remap_samples = np.array(csvr.readall(os.path.join(paths['remapDir'],val_img_names[v][:-3] + "csv"),csvr.READ_FLOAT))
			target = remap_samples[:,0]
			base_probs = remap_samples[:,1:]
			val_remap_targets.append(target)
			val_base_probs.append(base_probs)
			
	#more debugging
	if DEBUG:
		double_print('Example validation item',text_log)
		double_print(val_imgs[0],text_log)
		double_print(val_targets[0],text_log)
	
	#compute average of classes present for binary loss weights
	if not net_opts['is_target_distribution'] and net_opts['is_loss_weighted_by_class']:
		double_print('Computing class frequencies...',text_log)
		class_freq = class_frequency(paths['truth_dir'],net_opts['num_labels'],train_img_names)
		double_print(class_freq,text_log)
	else:
		class_freq = np.ones((1,net_opts['num_labels']),dtype=np.float32) #doesn't matter
	
	tf.reset_default_graph()
	
	if net_opts['is_gpu']:
		sess = tf.InteractiveSession()
	else:
		#debug to force CPU mode!
		config = tf.ConfigProto(
			device_count = {'GPU': 0}
		)
		sess = tf.InteractiveSession(config=config)
	
	#TODO: Move reg down into net when you make a network base class...
	network = net.PriorNet(net_opts)
	best_val_loss = tf.Variable(sys.float_info.max,trainable=False,name="best_val_loss")
	reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	total_loss = network.loss + reg_loss
	if DEBUG:
		double_print('Regularization Losses:',text_log)
		double_print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),text_log)
	
	train_op_handler = TrainOpHandler(net_opts,total_loss)
	
	#tensorboard inspection
	summaries = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(paths['checkpoint_dir'],graph=sess.graph)
	
	saver = tf.train.Saver()
	
	latest_model = get_latest_model(paths['checkpoint_dir'])
	if latest_model:
		tf.global_variables_initializer().run()
		saver.restore(sess,latest_model)
		start = int(latest_model.split('-')[-1])+1
		double_print('Starting from iteration %d. Current validation loss: %.5f' % (start, np.mean(sess.run([best_val_loss]))),text_log)
	else:
		double_print('Creating a new network...',text_log)
		start = 0
		tf.global_variables_initializer().run()
		network.load_weights(paths['weight_file'],sess)
	model_name = os.path.join(paths['checkpoint_dir'],paths['model_name'])
		
	trainable_vars = tf.trainable_variables()	
	if DEBUG:
		print("Trainable Variables:")
		print(trainable_vars)
	
	#TODO: Realistically, data fetching should be handled by its own class...any way to leave open option for tfrecord in doing so? Blegh...
	cur_train_indices = list(range(len(train_img_names)))
		
	#actual training loop
	for iter in range(start,net_opts['max_iter']):
				
		if iter == net_opts['iter_end_only_training']:
			double_print('Switching to end-to-end training.',text_log)
				
		if iter % (net_opts['batches_per_val_check']) == 0:
			#TODO: Encapsulate out all statistics you want somehow, so that changing the list doesn't change code in a dozen places
			val_loss = 0
			val_err = 0
			val_acc = 0
			for v_ind in range(len(val_img_names)):
				feed_dict = {network.inputs:val_imgs[v_ind],
							network.seg_target:val_targets[v_ind],
							network.class_frequency: class_freq,
							network.is_train: False}
							
				if net_opts['remapping_loss_weight'] > 0:
					feed_dict[network.remap_target] = val_remap_targets[v_ind]
					feed_dict[network.remap_base_prob] = val_base_probs[v_ind]
					feed_dict[network.map_mat] = map_mat
					
				loss, err, acc, smry = sess.run([network.loss, network.prior_err, network.seg_acc, summaries], feed_dict=feed_dict)
				val_loss += loss
				val_err += err
				val_acc += acc
			val_loss = np.mean(val_loss)/len(val_img_names)
			val_err = np.mean(val_err)/len(val_img_names)
			val_acc = np.mean(val_acc)/len(val_img_names)
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
			if new_best or iter % 10000 == 0:
				saver.save(sess, model_name,global_step = iter)
				
		batch_size = net_opts['batch_size']
			
		loss = 0
		err = 0
		acc = 0
		for item in range(batch_size):
			if len(cur_train_indices) == 0:
				cur_train_indices = list(range(len(train_img_names)))		
			t_ind = cur_train_indices.pop(random.randint(0,len(cur_train_indices)-1))
			
			im_name = os.path.join(paths['im_dir'], train_img_names[t_ind])
			#print(im_name)
			img = cv2.imread(im_name)
			sz = img.shape
			truth = my_load_int_mat(os.path.join(paths['truth_dir'],train_img_names[t_ind][:-3] + 'mat'))			
			(img,truth) = augment_img(img,truth)
			#debug
			double_print('img,truth:',text_log)
			double_print(img,text_log)
			double_print(truth,text_log)
			
			feed_dict={
				network.inputs:img[np.newaxis,:,:,:],
				network.seg_target: truth[np.newaxis,:,:],
				network.class_frequency: class_freq,
				network.is_train: True}
				
			if net_opts['remapping_loss_weight'] > 0:
				remap_samples = np.array(csvr.readall(os.path.join(paths['presoftmaxDir'],train_img_names[t_ind][:-3] + "csv")),csvr.READ_FLOAT)
				truth = remapSamples[:,0]
				base_probs = remapSamples[:,1:]
				feed_dict[network.remap_target] = remap_samples
				feed_dict[network.remap_base_prob] = base_probs
				feed_dict[network.map_mat_file] = map_mat
							
			_, cur_loss, cur_err,cur_acc = sess.run([train_op_handler.train_op(iter),network.loss,network.prior_err,network.seg_acc], feed_dict=feed_dict)
			
			loss += cur_loss
			err += cur_err
			acc += cur_acc
			
		train_op_handler.check_gradients(iter,sess)
		sess.run(train_op_handler.post_batch_op(iter))
		
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
