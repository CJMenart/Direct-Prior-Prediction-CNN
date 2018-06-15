"Loads from datasets saved as tfrecord, loading metadata from the matfiles created for CVL 5/2018.\
Should be faster than the non-tfrecord version, but technically offers less control. And requires you to make a tfrecord.\
Much jiggering was required to make this interchangeable with feed dictionaries."

#ALSO NOTE: There are 'newer' ways to do what I'm doing, but I have to work with Tensorflow 1.4 on HPC so that's commented out.

from data_loader import *
from my_read_mat import *
import scipy.io as sio
import os
import partition_enum as pe
import tensorflow as tf
import random

DEBUG = True
DAT_TYPE = tf.float32


class CVL2018TFRecordDataLoader(IDataLoader):

	def __init__(self, base_fcn_weight_dir, dataset_dir):
		self._base_fcn_weight_dir = base_fcn_weight_dir
		self._dataset_dir = dataset_dir
		(self._num_test,self._num_train,self._num_val,self._num_labels) = read_matfile(os.path.join(dataset_dir,'dataset_info.mat'),['num_test','num_train','num_val','num_labels'])
		
		self._batch_size = tf.placeholder(tf.int32,None)
		self._partition = tf.placeholder(tf.int32,None)
		def _parse_function(example_proto):
			features = {'image': tf.VarLenFeature((),tf.string,default_value=""),
						'label': tf.VarLenFeature((),tf.string,default_value=""),
						'height': tf.train.FixedLenFeature((),tf.int64,default_value=""),
						'width': tf.train.FixedLenFeature((),tf.int64,default_value="")}			
			parsed_features = tf.parse_single_example(example_proto,features)
			image = tf.decode_raw(parsed_features['image'], tf.int32)
			label = tf.decode_raw(parsed_features['label'], tf.int32)
			image = tf.reshape(image, [parsed_features['height'], parsed_features['width'], 3])
			return (image,label)

		self._img_splits = {}
		self._truth_splits = {}
		for split,splitname in zip([pe.TRAIN,pe.TEST,pe.VAL],['Train','Test','Val']):
			tfrecord = tf.data.TFRecordDataset(os.path.join(dataset_dir,'TFRecords','%s.tfrecord' % splitname))
			dataset = tfrecord.map(_parse_function)
			dataset = dataset.shuffle(buffer_size=1000)
			dataset = dataset.batch(self._batch_size)
			dataset = dataset.repeat()
			iterator = dataset.make_one_shot_iterator()
			self._img_splits[split], self._truth_splits[split] = iterator.get_next()
			
		self._img = tf.case({self._partition == pe.TRAIN: self._img_splits[pe.TRAIN],
							self._partition == pe.TEST: self._img_splits[pe.TEST],
							self._partition == pe.VAL: self._img_splits[pe.VAL]})
		self._truth = tf.case({self._partition == pe.TRAIN: self._truth_splits[pe.TRAIN],
							self._partition == pe.TEST: self._truth_splits[pe.TEST],
							self._partition == pe.VAL: self._truth_splits[pe.VAL]})
							
	def inputs(self):
		return self._img
	
	def seg_target(self):
		return self._truth
	
	#function where the magic happens in actual training
	def feed_dict(self,partition,batch_size):
		return {self._batch_size:batch_size,self._partition:partition}
		
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
		
