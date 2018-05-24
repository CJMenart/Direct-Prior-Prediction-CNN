" Marshalls all the parameters and settings to launch prior prediction, particularly as a batch job on high performance clusters."
import prior_prediction_learning as net
import sys
import os
import numpy as np
import argparse
from cvl_2018_data_loader import *

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
	parser.add_argument("--iter_end_only_training",type=int,default=1200)
	parser.add_argument("--is_clipped_gradients",type=int,default=False)
	parser.add_argument("--remapping_loss_weight",type=float,default=0)
	parser.add_argument("--batches_per_val_check",type=int,default=50)
	parser.add_argument("--is_batchnorm_fixed",type=int,default=False)
	parser.add_argument("--dropout_prob",type=float,default=0.5)
	parser.add_argument("--hid_layer_width",type=int,default=2048)
	parser.add_argument("--num_hid_layers",type=int,default=1)
	parser.add_argument("--is_target_distribution",type=int,default=False)
	parser.add_argument("--is_loss_weighted_by_class",type=int,default=False)
	parser.add_argument("--base_net",type=str,default='resnet_v2')
	parser.add_argument("--is_gpu",type=int,default=True)
	parser.add_argument("--fcn_weight_file",type=str,default='_')
	
	#path stuff--something here will probably be required
	parser.add_argument("--data_loader_type",type=str,default="CVL_2018")
	parser.add_argument("--base_fcn_weight_dir",type=str,default='_')
	parser.add_argument("--dataset_dir",type=str,default='_')
	
	args = parser.parse_args()
	net_opts = {}

	if args.data_loader_type == 'CVL_2018':
		data_loader = CVL2018DataLoader(args.base_fcn_weight_dir,args.dataset_dir)	
	else:
		raise Exception("data_loader type not recognized.")
		
	net_opts['model_name'] = 'DirectPriorNet'	
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
	net_opts['is_target_distribution'] = args.is_target_distribution
	net_opts['is_loss_weighted_by_class'] = args.is_loss_weighted_by_class
	net_opts['base_net'] = args.base_net
	net_opts['is_gpu'] = args.is_gpu
	net_opts['err_thresh'] = 1e-3
	net_opts['fcn_weight_file'] = args.fcn_weight_file
	net_opts['iter_per_automatic_backup'] = 10000
	
	#currently unused, may implement later
	#net_opts['is_im_size_fixed'] = False
	
	'''
	if args.dataset == 'MS_COCO':
		data_loader.num_labels() = 90
	elif args.dataset == 'PASCAL_Context':
		data_loader.num_labels() = 59
	elif args.dataset == 'ADE20K':
		data_loader.num_labels() = 150
	elif args.dataset == 'NYUDv2':
		data_loader.num_labels() = 40
	else:
		print('NetTest: Error: unrecognized dataset name')
	'''
	
	#TODO probably will neeed more switch cases in future
	if args.is_eval_mode:
		net.testing(paths,net_opts)
	else:
		net.training(net_opts,data_loader,args.checkpoint_dir)