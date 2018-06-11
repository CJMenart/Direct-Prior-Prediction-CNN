"Loads from datasets saved as matfiles, format created for CVL 5/2018.\
This setup basically incompatible with TFRecords. But we're not in production code here, this is quickly-changing research and I am DONE with tfrecord's inflexibility atm."
from data_loader import *
from my_read_mat import *
import scipy.io as sio
import os
import partition_enum
import tensorflow as tf
import random

#TODO if we want to get fancy we could also scan the actual filesize of the folder
MAX_VAL_PRELOAD = 250
DEBUG = True
DAT_TYPE = tf.float32


class CVL2018DataLoader(IDataLoader):

	def __init__(self, base_fcn_weight_dir, dataset_dir):
		self._base_fcn_weight_dir = base_fcn_weight_dir
		self._dataset_dir = dataset_dir
	
		(self._num_test,self._num_train,self._num_val,self._num_labels) = read_matfile(os.path.join(dataset_dir,'dataset_info.mat'),['num_test','num_train','num_val','num_labels'])
		
		if self._num_val < MAX_VAL_PRELOAD:
			self._preloaded_val = False
			self._val_imgs = []
			self._val_truth = []
			for im in range(self._num_val):
				(img,truth) = val_img_and_truth(im)
				self._val_imgs.append(img)
				self._val_truth.append(truth)
			self._preloaded_val = True
		else: 
			self._preloaded_val = False
				
		self._inputs = tf.placeholder(DAT_TYPE,[None,None,None,3])
		self._seg_target = tf.placeholder(tf.int64,[None,None,None]) #batch,width,height
	
		self._cur_epoch_indices = {}
		for part in [partition_enum.TRAIN,partition_enum.TEST,partition_enum.VAL]:
			self._cur_epoch_indices[part] = []
	
	def inputs(self):
		return self._inputs
	
	def seg_target(self):
		return self._seg_target
	
	#function where the magic happens in actual training
	def feed_dict(self,partition,batch_size):
		imgs = []
		truths = []
		for i in range(batch_size):
			img,truth = self.img_and_truth(self._get_epoch_index(partition),partition)
			imgs.append(img)
			truths.append(truth)
		return {self._inputs:imgs,self._seg_target:truths}
	
	def base_fcn_weight_dir(self):
		return self._base_fcn_weight_dir

	#preserved for legacy--variety of tests and utilities take advantage of power to pull out data directly
	#so we just took advantage of it for feed_dict b/c it made sense
	def img_and_truth(self,ind,partition):
		if partition == partition_enum.TEST:
			assert ind <= self._num_test
			return self._img_and_truth(ind,'TestImgs','test')
		elif partition == partition_enum.VAL:
			assert ind <= self._num_val
			if self._preloaded_val:
				return (self._val_imgs[ind],self._val_truth[ind])
			else:
				return self._img_and_truth(ind,'ValImgs','val')
		elif partition == partition_enum.TRAIN:
			assert ind <= self._num_train
			return self._img_and_truth(ind,'TrainImgs','train')
		else:
			raise Exception('Unrecognized partition.')
			
	def num_labels(self):
		return self._num_labels
		
	def num_data_items(self,partition):
		if partition == partition_enum.TRAIN:
			return self._num_train
		if partition == partition_enum.TEST:
			return self._num_test
		if partition == partition_enum.VAL:
			return self._num_val
			
	#we have not implemented methods for remapping here
	def semantic_presoftmax(self,ind,partition):
		raise NotImplementedError
			
	def semantic_prob(self,ind,partition):
		raise NotImplementedError
					
	def map_mat(self):
		raise NotImplementedError
		
	#TODO: Name too similar to another function
	def _img_and_truth(self,ind,subfolder,prefix):
		img_fname = os.path.join(self._dataset_dir,subfolder,'%s_%06d_img.png' % (prefix,ind+1))
		img = cv2.imread(img_fname)
		truth_fname = os.path.join(self._dataset_dir,subfolder,'%s_%06d_pixeltruth.mat' % (prefix,ind+1))
		truth = read_matfile(truth_fname,'truth_img')
		return (img,truth)
		
	def _get_epoch_index(self,partition):
		if len(self._cur_epoch_indices[partition]) == 0:
			if DEBUG:
				print('re-filling epoch indices.')
			self._cur_epoch_indices[partition] = list(range(self.num_data_items(partition)))
		return self._cur_epoch_indices[partition].pop(random.randint(0,len(self._cur_epoch_indices[partition])-1))
		