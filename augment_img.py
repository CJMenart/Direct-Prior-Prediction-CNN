"set of basic augmentations for any image processing with a CNN. \
NOTE: You are required to call augmentation methods here before any changing of the image range out of the 0-255 space. Otherwise the gamma-scaling works improperly."

import numpy as np
import cv2
import os
from img_util import resize_ratio
import cvl_2018_data_loader as loading

GAMMA_RANGE = [0.5,2]
AREA_RANGE = [200*200,600*600] 
LB_SCALE_SET = [0.7,0.8,0.9,1.0,1.1,1.2,1.3]


def augment_no_size_change(img,truth=None):
	"augments by flipping, gamma scaling."
	
	#flipping
	if np.random.randint(0,2) == 0:
		img = np.fliplr(img)
		if truth is not None:
			truth = np.fliplr(truth)
		
	#lighting augmentation
	img = gamma_correction(img,np.random.uniform(GAMMA_RANGE[0],GAMMA_RANGE[1]))

	return (img,truth)


def augment_LB(img,truth=None):
	"augment in the style of LabelBank paper by flipping and scaling.\
	ground-truth 1-channel image will also be augmented to match if passed in."

	#flipping
	if np.random.randint(0,2) == 0:
		img = np.fliplr(img)
		if truth is not None:
			truth = np.fliplr(truth)
			
	#scaling
	scale = LB_SCALE_SET[np.random.randint(0,len(LB_SCALE_SET))]
	(img,truth) = resize_ratio(img,scale,truth)
	return (img,truth)

	
def augment_img(img,truth=None,fixed_sz=None):
	"augments by flipping, scaling, gamma scaling.\
	ground-truth 1-channel image will also be augmented to match if passed in.\
	if fixed_sz is set, image will be resized to the size passed in "

	#flipping
	if np.random.randint(0,2) == 0:
		img = np.fliplr(img)
		if truth is not None:
			truth = np.fliplr(truth)
		
	if fixed_sz:
		img = cv2.resize(img, (fixed_sz, fixed_sz))
	else:
		#randomly resize
		#WARNING: No checks done to ensure length of smallest side
		size = img.shape
		area = size[0]*size[1]
		(img,truth) = resize_ratio(img,np.sqrt(np.random.randint(AREA_RANGE[0],AREA_RANGE[1])/area),truth)
		#scale = LB_SCALE_SET[np.random.randint(0,len(LB_SCALE_SET))]
		#(img,truth) = resize_ratio(img,scale,truth) #commented-out produces mysterious errors but only on HPC :(
		
	#lighting augmentation
	img = gamma_correction(img,np.random.uniform(GAMMA_RANGE[0],GAMMA_RANGE[1]))

	return (img,truth)

	
def test_aug(dataset_dir):
	import tensorflow as tf
	import augment_img_node as aug_node
	import partition_enum
	from matplotlib import pyplot as plt
	"Quick effort to test tensorflow-ified augmentation."
	sess = tf.InteractiveSession()
	net_opts = {'img_sizing_method': 'pad_input','standard_image_size':[481,481],'batch_size':1,'base_fcn_weight_dir':'_','dataset_dir':dataset_dir}
	loader = loading.CVL2018DataLoader(net_opts)
	img_in = loader.inputs()
	truth_in = loader.seg_target()
	aug_img, aug_truth = aug_node.augment_no_size_change(img_in,truth_in)
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
	
	
def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, 1/correction)
    return np.uint8(img*255)
	
	
def test_augment(img_fname,truth_fname,save_dir):
	img = cv2.imread(img_fname)
	truth = np.array(csvr.readall(truth_fname,csvr.READ_INT))
	for trial in range(20):
		aug_img,aug_truth = augment_img(img,truth)
		cv2.imwrite(os.path.join(save_dir,'aug_img%03d.png' % trial),aug_img)
		cv2.imwrite(os.path.join(save_dir,'aug_truth%03d.png' % trial),aug_truth)
	for trial in range(20):
		aug_img,aug_truth = augment_img(img,truth,100)
		cv2.imwrite(os.path.join(save_dir,'aug_img_fixed%03d.png' % trial),aug_img)
		cv2.imwrite(os.path.join(save_dir,'aug_truth_fixed%03d.png' % trial),aug_truth)