""" Marshalls all the parameters and settings to launch prior prediction, particularly as a batch job on high performance clusters."""
import prior_prediction_learning as net
import sys
import os
import numpy as np
import argparse

#There is one required positional paramter, the directory to save checkpoints in.
#NOTE: intean command-line arguments are integers here--there's no great way to do it with argparse.
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("checkpoint_dir")
	parser.add_argument("--regularization_weight",type=float,default=1e-5)
	parser.add_argument("--dataset",type=str,default="PASCAL_Context")
	parser.add_argument("--optimizer_type",default='Adam')
	parser.add_argument("--momentum",type=float,default=0.99)
	parser.add_argument("--is_using_nesterov",type=int,default=False)
	parser.add_argument("--batch_size",type=int,default=1)
	parser.add_argument("--learn_rate",type=float,default=1e-3)
	parser.add_argument("--is_eval_mode",type=int,default=False)
	parser.add_argument("--max_iter",type=int,default=1000000)
	parser.add_argument("--is_custom_paths",type=int,default=False)
	parser.add_argument("--weight_file",type=str,default=None)
	parser.add_argument("--train_name_file",type=str,default=None)
	parser.add_argument("--test_name_file",type=str,default=None)
	parser.add_argument("--val_name_file",type=str,default=None)
	parser.add_argument("--im_dir",type=str,default=None)
	parser.add_argument("--truth_dir",type=str,default=None)
	parser.add_argument("--iter_end_only_training",type=int,default=1200)
	parser.add_argument("--is_clipped_gradients",type=int,default=False)
	parser.add_argument("--remapping_loss_weight",type=float,default=0)
	parser.add_argument("--presoftmax_dir",type=str,default=None)
	parser.add_argument("--batches_per_val_check",type=int,default=50)
	parser.add_argument("--is_batchnorm_fixed",type=int,default=False)
	parser.add_argument("--dropout_prob",type=float,default=0.5)
	parser.add_argument("--hid_layer_width",type=int,default=2048)
	parser.add_argument("--num_hid_layers",type=int,default=1)
	parser.add_argument("--is_target_distribution",type=int,default=False)
	parser.add_argument("--is_loss_weighted_by_class",type=int,default=False)
	parser.add_argument("--map_mat_file",type=str,default=None)
	parser.add_argument("--train_clusters_file",type=str,default=None)
	parser.add_argument("--val_clusters_file",type=str,default=None)
	parser.add_argument("--base_net",type=str,default='resnet_v2')
	parser.add_argument("--is_gpu",type=int,default=True)
	
	args = parser.parse_args()
	
	paths = {}
	net_opts = {}

	if args.mobilePaths:
		paths['checkpoint_dir'] = args.checkpoint_dir
		paths['train_name_file'] = args.train_name_file
		paths['weight_file'] = args.weight_file
		paths['test_name_file'] = args.test_name_file
		paths['im_dir'] = args.im_dir
		paths['truth_dir'] = args.truth_dir
		paths['presoftmax_dir'] = args.presoftmax_dir
		paths['map_mat_file'] = args.map_mat_file
	else:
		base_dir = '/p/work1/workspace/cmenart/'
		paths['checkpoint_dir'] = os.path.join(base_dir,args.dataset,args.checkpoint_dir)
		paths['test_name_file'] = os.path.join(base_dir,'Prior Classification ' + args.dataset,'Testing Data','prior_test_img_names.csv')
		paths['train_name_file'] = os.path.join(base_dir,'Prior Classification ' + args.dataset,'Training Data','prior_train_img_names.csv')
		paths['val_name_file'] = os.path.join(base_dir,'Prior Classification ' + args.dataset,'Training Data','prior_val_img_names.csv')
		if args.weight_file:
			paths['weight_file'] = os.path.join(base_dir,'Prior Classification ' + args.dataset,args.weight_file) 
		else:
			paths['weight_file'] = None
		paths['train_name_file'] = os.path.join(base_dir,'Prior Classification ' + args.dataset,'Training Data','prior_train_img_names.csv')
		paths['im_dir'] = os.path.join(base_dir, args.dataset, 'Images/')
		paths['truth_dir'] = os.path.join(base_dir,args.dataset,'Ground Truth CSV/')
		paths['presoftmax_dir'] = os.path.join(base_dir,args.dataset,'Presoftmax CSV/')
		paths['train_clusters_file'] = os.path.join(base_dir,'Prior Classification ' + args.dataset,'Training Data','train_clustering.csv')
		paths['val_clusters_file'] = os.path.join(base_dir,'Prior Classification ' + args.dataset,'Testing Data','val_clustering.csv')
		paths['map_mat_file'] = 'TODO'
		paths['model_name'] = 'DirectPriorNet'
		
	net_opts['batches_per_val_check'] = args.batches_per_val_check
	net_opts['regularization_weight'] = args.regularization_weight
	net_opts['optimizer_type'] = args.optimizer_type
	net_opts['momentum'] = args.momentum
	net_opts['batch_size'] = args.batch_size
	net_opts['epsilon'] = 1e-12
	net_opts['adam_epsilon'] = 1e-4
	net_opts['learn_rate'] = args.learn_rate
	net_opts['max_iter'] = args.max_iter
	net_opts['is_using_nesterov'] = args.is_using_nesterov
	net_opts['iter_end_only_training'] = args.iter_end_only_training
	net_opts['is_clipped_gradients'] = args.is_clipped_gradients
	net_opts['remapping_loss_weight'] = args.remapping_loss_weight
	net_opts['is_batchnorm_fixed'] = args.is_batchnorm_fixed
	net_opts['dropout_prob'] = args.dropout_prob
	net_opts['hid_layer_width'] = args.hid_layer_width
	net_opts['num_hid_layers'] = args.num_hid_layers
	net_opts['is_distribution'] = args.is_distribution
	net_opts['is_loss_weighted_by_class'] = args.is_loss_weighted_by_class
	net_opts['base_net'] = args.base_net
	net_opts['is_gpu'] = args.is_gpu
	
	#currently unused, may implement later
	net_opts['is_im_size_fixed'] = False
	
	if args.dataset == 'MS_COCO':
		net_opts['num_labels'] = 90
	elif args.dataset == 'PASCAL_Context':
		net_opts['num_labels'] = 59
	elif args.dataset == 'ADE20K':
		net_opts['num_labels'] = 150
	elif args.dataset == 'NYUDv2':
		net_opts['num_labels'] = 40
	else:
		print('NetTest: Error: unrecognized dataset name')
		
	#TODO probably will neeed more switch cases in future
	if args.evalMode:
		net.testing(paths,net_opts)
	else:
		net.train_on_all(paths,net_opts)