"Build feed dictionary for prior_net given data_loader, network, and indices."

import numpy as np
import partition_enum
from augment_img import *
import cv2
from img_util import *

DEBUG = False

#IMPORTANT: If we ever break out prior_net into interface and implementation, this should be coupled only to the interface. Obvs.


def build_feed_dict(data_loader,network,inds,partition,net_opts):
	#containers for input data
	imgs = []
	truths = []
	semantic_probs = []

	for ind in inds:
		
		img,truth = data_loader.img_and_truth(ind,partition)
		if partition == partition_enum.TRAIN:
			(img,truth) = augment_no_size_change(img,truth) 
		if DEBUG:
			double_print('img,truth:',text_log)
			double_print(img,text_log)
			double_print(truth,text_log)		
		imgs.append(img)
		truths.append(truth)
		
		if net_opts['remapping_loss_weight'] > 0:
			semantic_probs.append(data_loader.semantic_probs(ind,partition))
		
	imgs,truths = _size_imgs(imgs,truths,net_opts)
	feed_dict={
		network.inputs: imgs,
		network.seg_target: truths,
		network.is_train: partition == partition_enum.TRAIN}

	if net_opts['is_loss_weighted_by_class']:
		feed_dict[network.class_frequency] = data_loader.class_frequency(),
	if net_opts['remapping_loss_weight'] > 0:
		feed_dict[network.remap_target] = truths #prob wrong need to change when we actually want to use this
		feed_dict[network.remap_base_prob] = remap_base_probs
		feed_dict[network.map_mat] = data_loader.map_mat()
		
	return feed_dict
			
			
def _size_imgs(imgs,truths,net_opts):
	"Size-related preprocessing." 
	if len(imgs) == 1:
		return (imgs,truths) #no need to alter single image
	elif net_opts['img_sizing_method'] == 'standard_size':
		for i in range(len(imgs)):
			imgs[i] = cv2.resize(imgs[i], (net_opts['standard_image_size'][0],net_opts['standard_image_size'][1]))
			truths[i] = cv2.resize(truths[i], (net_opts['standard_image_size'][0],net_opts['standard_image_size'][1]),interpolation=cv2.INTER_NEAREST)
	elif net_opts['img_sizing_method'] == 'pad_input':
		pad_size = net_opts['standard_size']
		for i in range(len(imgs)):
			ratio = np.asscalar(np.min(np.array(pad_size)/imgs[i].shape[:2]))
			if DEBUG:
				print('resize ratio for padded input:')
				print(ratio)
			imgs[i],truths[i] = resize_ratio(imgs[i],ratio,truths[i])
			imgs[i],truths[i] = pad_to_size(imgs[i],pad_size,truths[i])
	else:
		raise Exception('Not sure how to handle image size.')
	
	return imgs,truths