import matplotlib.pyplot as plt
from cvl_2018_data_loader import *
import build_feed_dict
import partition_enum
import augment_img
import time


def test_img_preprocessing(dataset_dir):
	"Make sure that our preprocessed images come out ok by looking at them."
	data_loader = CVL2018DataLoader('_',dataset_dir)
	imgs = []
	truths = []
	batch_sz = 2
	for i in range(batch_sz):
		img,truth = data_loader.img_and_truth(i+100,partition_enum.TRAIN)	
		imgs.append(img)
		truths.append(truth)
	_show_me_imgs(imgs,truths)
	for i in range(batch_sz):
		(imgs[i],truths[i]) = augment_img.augment_no_size_change(imgs[i],truths[i])
	_show_me_imgs(imgs,truths)
		
	net_opts = {}
	net_opts['standard_image_size'] = [256,256]
	
	'''
	net_opts['img_sizing_method'] = 'standard_size'
	imgsA,truthsA = build_feed_dict._size_imgs(imgs,truths,net_opts)
	_show_me_imgs(imgsA,truthsA)
	'''
	net_opts['img_sizing_method'] = 'pad_input'
	imgsB,truthsB = build_feed_dict._size_imgs(imgs,truths,net_opts)
	_show_me_imgs(imgsB,truthsB)
	
	
def _show_me_imgs(imgs,truths):
	for i in range(len(imgs)):
		print(imgs[i].shape)
		plt.imshow(imgs[i])
		plt.show()
		print(truths[i].shape)
		plt.imshow(truths[i])
		plt.show()
