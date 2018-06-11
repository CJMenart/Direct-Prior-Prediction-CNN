"Loads from datasets saved as tfrecord, loading metadata from the matfiles created for CVL 5/2018.\
Should be faster than the non-tfrecord version, but technically offers less control. And requires you to make a tfrecord."

from data_loader import *
from my_read_mat import *
import scipy.io as sio
import os
import partition_enum
import tensorflow as tf
import random

DEBUG = True
DAT_TYPE = tf.float32

class CVL2018DataLoader(IDataLoader):

	def __init__(self, base_fcn_weight_dir, dataset_dir):
		self._base_fcn_weight_dir = base_fcn_weight_dir
		self._dataset_dir = dataset_dir
	
		(self._num_test,self._num_train,self._num_val,self._num_labels) = read_matfile(os.path.join(dataset_dir,'dataset_info.mat'),['num_test','num_train','num_val','num_labels'])
		
		
		def _parse_function(example_proto):
			features = {'image':  ,
						'label':
						'height':
						'width':}
			parsed_features = tf.parse_single_example(example_proto,features)
			
			
			
		
		for split,splitname in zip([pe.TRAIN,pe.TEST,pe.VAL],['Train','Test','Val']):
			tfrecord = tf.data.TFRecordDataset(os.path.join(dataset_dir,'%s.tfrecrd' % splitname))
			decoded_dataset = tfrecord.map(_parse_function)
			
		
		self._inputs = tf.placeholder(DAT_TYPE,[None,None,None,3])
		self._seg_target = tf.placeholder(tf.int64,[None,None,None]) #batch,width,height
	

	
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
		
