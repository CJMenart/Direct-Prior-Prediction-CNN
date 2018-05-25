"Utility functions for working with image/truth pairs."

import cv2
import numpy as np

DEBUG = False


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