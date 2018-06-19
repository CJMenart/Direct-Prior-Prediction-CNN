"set of basic augmentations for any image processing with a CNN. Done all in TF Nodes, as opposed to outdated augment_img. May be called on a stack of images.\
NOTE: You are required to call augmentation methods here before any changing of the image range out of the 0-255 space. Otherwise the gamma-scaling works improperly."

import numpy as np
import os
import partition_enum
from matplotlib import pyplot as plt
import cvl_2018_tfrecord_data_loader as loading
import tensorflow as tf

GAMMA_RANGE = [0.5,2]
AREA_RANGE = [200*200,600*600] 
LB_SCALE_SET = [0.7,0.8,0.9,1.0,1.1,1.2,1.3]

def augment_no_size_change(imgs,truth=None):
	"augments by flipping, gamma scaling."
		
	def random_flip_left_right_pair(pair):
		img,truth = pair
		return tf.cond(tf.equal(1.0,tf.round(tf.random_uniform([],0,1))),
			lambda: (img[:,::-1,:],truth[:,::-1]), lambda: (img,truth))
	
	#randomly flip images
	if truth is None:
		imgs = tf.map_fn(tf.image.random_flip_left_right, imgs)
	else:
		imgs,truth = tf.map_fn(random_flip_left_right_pair, (imgs,truth))
			
	#lighting augmentation
	imgs = _gamma_correction(imgs,tf.random_uniform([],GAMMA_RANGE[0],GAMMA_RANGE[1]))
	
	return (imgs,truth)

	
def _gamma_correction(img, correction):
    img = img/tf.constant(255.0,dtype=tf.float32)
    img = tf.pow(img, tf.constant(1.0,dtype=tf.float32)/correction)
    img = img*255.0
    return img
	
	
def test_aug(dataset_dir):
	"Quick effort to test tensorflow-ified augmentation."
	sess = tf.InteractiveSession()
	net_opts = {'img_sizing_method': 'pad_input','standard_image_size':[481,481],'batch_size':1,'base_fcn_weight_dir':'_','dataset_dir':dataset_dir}
	loader = loading.CVL2018TFRecordDataLoader(net_opts)
	img_in = loader.inputs()
	truth_in = loader.seg_target()
	aug_img, aug_truth = augment_no_size_change(img_in,truth_in)
	#(aug_img, aug_truth) = size_imgs(aug_img,aug_truth,net_opts)	
	
	tf.global_variables_initializer().run()
	split = partition_enum.TRAIN
	#saver.restore(sess,weight_fname)
	for i in range(loader.num_data_items(split)):
		feed_dict = loader.feed_dict(partition_enum.TRAIN,1)
		
		img_before,truth_before,img_after,truth_after = sess.run([img_in,truth_in,aug_img,aug_truth],feed_dict=feed_dict)
		
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		ax1.imshow(np.squeeze(img_before[0,:,:,:]/255))
		ax2 = fig.add_subplot(212)
		ax2.imshow(np.squeeze(truth_before[0,:,:]))
		plt.show()
				
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		ax1.imshow(np.squeeze(img_after[0,:,:,:]/255))
		ax2 = fig.add_subplot(212)
		ax2.imshow(np.squeeze(truth_after[0,:,:]))
		plt.show()
		
def size_imgs(imgs,truths,net_opts):
	"Size-related preprocessing." 
	#Technically no action is needed if there's only one image. But we don't turn it off in that case. I feel this is the best way to avoid unexpected behavior.
	if net_opts['img_sizing_method'] == 'run_img_by_img':
		return (imgs,truths) #no need to alter single image
	elif net_opts['img_sizing_method'] == 'standard_size':
		imgs = tf.image.resize_images(imgs,net_opts['standard_image_size'],align_corners=True,method=tf.image.ResizeMethod.BICUBIC)
		truths = tf.squeeze(tf.image.resize_images(tf.expand_dims(truths,3),net_opts['standard_image_size'],align_corners=True,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), axis=3)
		return (imgs,truths)
	elif net_opts['img_sizing_method'] == 'pad_input':
	
		imgs = tf.image.resize_image_with_crop_or_pad(imgs,*net_opts['standard_image_size'])
		truths = tf.squeeze(tf.image.resize_image_with_crop_or_pad(tf.expand_dims(truths,3),*net_opts['standard_image_size']),axis=3)
		return (imgs,truths)
	else:
		raise Exception('Not sure how to handle image size.')
		

		
