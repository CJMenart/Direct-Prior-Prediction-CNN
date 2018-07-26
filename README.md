Direct Prior Prediction CNN

This python project trains a neural network which takes images and attempts to predict the distribution of object classes in the image (or the presence/absence of classes) as given by semantic segmentation labels for full supervision. Tested in Tensorflow 1.4.

Required packages: Tensorflow (1.4 or newer probably?), opencv, scipy, pyplot, h5py

To get started, check out prior_prediction_learning, prior_net, and prior_prediction_main.

A good starting command to train your own network is:

python prior_prediction_main.py NET_FOLDER --dataset PASCAL_Context --optimizer_type Momentum --momentum 0.5 --batch_size 32 --learn_rate 5e-1 --max_iter 20000 --iter_end_only_training 99999 --is_batchnorm_fixed 1 --hid_layer_width 1024 --num_hid_layers 1 --fcn_weight_file resnet_v1_152.ckpt --img_sizing_method standard_size --is_fc_batchnorm 1 --base_fcn_pooling_mode avg --fcn_pool_sz_a 1 --fcn_pool_sz_b 3 --fcn_pool_sz_c 6 --pyramid_pool_dim 256 --base_fcn_weight_dir PATH_TO_RESNET_WEIGHTS --dataset_dir PATH_TO_PASCAL_CONTEXT --is_target_distribution 0 --dropout_prob 0.2 --regularization_weight 1e-6 