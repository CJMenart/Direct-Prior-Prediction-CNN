"Utility functions for working with image/truth pairs."

import cv2
import numpy as np
import tensorflow as tf

DEBUG = False


def size_imgs(imgs,truths,net_opts):
	"size-related processing with cv2."
	#Technically no action is needed if there's only one image. But we don't turn it off in that case. I feel this is the best way to avoid unexpected behavior.
	if net_opts['img_sizing_method'] == 'run_img_by_img':
		return (imgs,truths) #no need to alter single image
	elif net_opts['img_sizing_method'] == 'standard_size':
		for i in range(len(imgs)):
			imgs[i] = cv2.resize(imgs[i], (fixed_sz, fixed_sz))
			truths[i] = cv2.resize(truths[i],(fixed_sz,fixed_sz),interpolation=cv2.INTER_NEAREST)
		return (imgs,truths)
	elif net_opts['img_sizing_method'] == 'pad_input':
		for i in range(len(imgs)):
			std_sz = net_opts['standard_image_size']
			min_ratio = min(std_sz[0]/imgs[i].shape[0],std_sz[1]/imgs[i].shape[1])
			img,truth = resize_ratio(imgs[i],min_ratio,truths[i])
			imgs[i],truths[i] = pad_to_size(img,std_sz,truth)
		return (imgs,truths)
	else:
		raise Exception('Not sure how to handle image size.')


def size_imgs_node(imgs,truths,net_opts):
	"Size-related preprocessing as tensorflow subgraph." 
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


def resize_ratio(img,ratio,truth=None):
	"Scale image up by a ratio without changing aspect ratio."
	size = img.shape
	if DEBUG:
		print('types in resize_ratio:')
		print(type(ratio))
		print(type(size[0]))
	img = cv2.resize(img, ( int(round(size[1]*ratio)), int(round(size[0]*ratio))))
	if truth is not None:
		truth = cv2.resize(truth,(int(round(size[1]*ratio)), int(round(size[0]*ratio))),interpolation=cv2.INTER_NEAREST)
	return (img,truth)
	

def pad_to_size(img,sz,truth=None):
	img_sz = img.shape
	extra_height = sz[0] - img_sz[0]
	extra_width = sz[1] - img_sz[1]
	assert(extra_height >= 0)
	assert(extra_width >= 0)
	img = cv2.copyMakeBorder(img,int(np.floor(extra_height/2)),int(np.ceil(extra_height/2)),\
		int(np.floor(extra_width/2)),int(np.ceil(extra_width/2)),cv2.BORDER_CONSTANT,value=[0,0,0])
	if truth is not None:
		truth = cv2.copyMakeBorder(truth,int(np.floor(extra_height/2)),int(np.ceil(extra_height/2)),\
			int(np.floor(extra_width/2)),int(np.ceil(extra_width/2)),cv2.BORDER_CONSTANT,value=0)
	return (img,truth)