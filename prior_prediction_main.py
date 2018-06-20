" Marshalls all the parameters and settings to launch prior prediction, particularly as a batch job on high performance clusters."
import prior_prediction_learning as net
import sys
import os
import numpy as np
import argparse

#There is one required positional paramter, the directory to save checkpoints in.
#NOTE: intean command-line arguments are integers here--there's no great way to do it with argparse.
#TODO: Way to have lower learning rate for base fcn
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
	parser.add_argument("--base_net",type=str,default='resnet_v1_152')
	parser.add_argument("--is_gpu",type=int,default=True)
	parser.add_argument("--fcn_weight_file",type=str,default='_')
	parser.add_argument("--img_sizing_method",type=str,default='pad_input')
	parser.add_argument("--is_fc_batchnorm",type=int,default=False)
	
	#path stuff--something here will probably be required
	parser.add_argument("--data_loader_type",type=str,default="CVL_2018")
	parser.add_argument("--base_fcn_weight_dir",type=str,default='_')
	parser.add_argument("--dataset_dir",type=str,default='_')
	
	args = parser.parse_args()
	net_opts = {}

	#for data loading--I was trying to keep paths out of net_opts, but maybe this is fine?
	#was constructing data_loader here. But now it has graph element and so needs to be constructed after tf session... :(
	net_opts['data_loader_type'] = args.data_loader_type
	net_opts['base_fcn_weight_dir'] = args.base_fcn_weight_dir
	net_opts['dataset_dir'] = args.dataset_dir
	
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
	net_opts['is_fc_batchnorm'] = args.is_fc_batchnorm
	net_opts['iter_per_automatic_backup'] = 10000
	net_opts['num_dropout_eval_reps'] = 32
	#resnet works best for dense prediction with size C*32 + 1, according to code comments
	net_opts['standard_image_size'] = [32*15+1,32*15+1]
	
	net_opts['img_sizing_method'] = args.img_sizing_method
	assert(	net_opts['img_sizing_method'] == 'run_img_by_img' or 	#may use to avoid all possible im distortions for ims of different size, but slower
			net_opts['img_sizing_method'] == 'standard_size' or     #quickest and involves no padding, but may introduce distortions
			net_opts['img_sizing_method'] == 'pad_input')           #quicker than running image by image, but may mess with batch norm if aspect ratios are crazy different
			
	#TODO probably will neeed more switch cases in future
	if args.is_eval_mode:
		net.testing(paths,net_opts)
	else:
		net.training(net_opts,args.checkpoint_dir)