"set of basic augmentations for any image processing with a CNN. Done all in TF Nodes, as opposed to outdated augment_img. May be called on a stack of images.\
NOTE: You are required to call augmentation methods here before any changing of the image range out of the 0-255 space. Otherwise the gamma-scaling works improperly."

import numpy as np
import os
import partition_enum
from matplotlib import pyplot as plt
import cvl_2018_data_loader as loading
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
    img = img/255.0
    img = tf.pow(img, 1/correction)
    return img
	
	
def test_aug(dataset_dir):
	"Quick effort to test tensorflow-ified augmentation."
	img_in = tf.placeholder(tf.float32,[None,None,None,3])
	truth_in = tf.placeholder(tf.int64,[None,None,None])
	aug_img, aug_truth = augment_no_size_change(img_in,truth_in)
	net_opts = {'img_sizing_method': 'standard_size','standard_image_size':[321,321]}
	(aug_img, aug_truth) = size_imgs(aug_img,aug_truth,net_opts)
	
	loader = loading.CVL2018DataLoader('_',dataset_dir)
	
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	split = partition_enum.TRAIN
	#saver.restore(sess,weight_fname)
	for i in range(loader.num_data_items(split)):
		(img,truth) = loader.img_and_truth(i,split)
		
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		ax1.imshow(img)
		ax2 = fig.add_subplot(212)
		ax2.imshow(truth)
		plt.show()
		
		img,truth = sess.run([aug_img,aug_truth],feed_dict={img_in:img[np.newaxis,:,:,:],truth_in:truth[np.newaxis,:,:]})
		
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		ax1.imshow(np.squeeze(img[0,:,:,:]))
		ax2 = fig.add_subplot(212)
		ax2.imshow(np.squeeze(truth[0,:,:]))
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
		raise NotImplementedError
		
		#TODO get augmentation and sizing back into actual pipeline...
		pad_size = net_opts['standard_image_size']
		for i in range(len(imgs)):
			ratio = np.asscalar(np.min(np.array(pad_size)/imgs[i].shape[:2]))
			if DEBUG:
				print('resize ratio for padded input:')
				print(ratio)
			imgs[i],truths[i] = resize_ratio(imgs[i],ratio,truths[i])
			imgs[i],truths[i] = pad_to_size(imgs[i],pad_size,truths[i])
	else:
		raise Exception('Not sure how to handle image size.')
		

		
