"Loads from datasets saved as tfrecord, loading metadata from the matfiles created for CVL 5/2018.\
Should be faster than the non-tfrecord version, but technically offers less control. And requires you to make a tfrecord.\
Much jiggering was required to make this interchangeable with feed dictionaries. Probably one of the biggest ninja moves in\
THe entire codebase, dependant upon every other part of the code working on an Inception-style pattern of TF use wherever possible."

#NOTE: Using functions unavailable in TF 1.4, not sure when they were added I upgraded straight to 1.8
#Also note that the TF stuff used below is very poorly documented and some trial-and-error was needed to figure it out. Changes may be hard.
#based largely on tutorial (for older version of TF) here:
#http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

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

	def __init__(self, base_fcn_weight_dir, dataset_dir, batch_size):
		self._base_fcn_weight_dir = base_fcn_weight_dir
		self._dataset_dir = dataset_dir
		(self._num_test,self._num_train,self._num_val,self._num_labels) = read_matfile(os.path.join(dataset_dir,'dataset_info.mat'),['num_test','num_train','num_val','num_labels'])
		
		self._batch_size = batch_size
		self._partition = tf.placeholder(tf.int64,None)
		def _parse_function(example_proto):
			features = {'image': tf.FixedLenFeature([],tf.string),
						'label': tf.FixedLenFeature([],tf.string),
						'height': tf.FixedLenFeature([],tf.int64),
						'width': tf.FixedLenFeature([],tf.int64)}			
			parsed_features = tf.parse_single_example(example_proto,features)
			#image = tf.decode_raw(parsed_features['image'], tf.uint8)
			image = tf.decode_raw(parsed_features['image'], tf.uint8)
			label = tf.decode_raw(parsed_features['label'], tf.int64)
			#width = tf.reshape( tf.decode_raw(parsed_features['width'], tf.int32),[])
			#height = tf.reshape( tf.decode_raw(parsed_features['height'], tf.int32),[])
			width = parsed_features['width']
			height = parsed_features['height']
			image = tf.reshape(image, tf.stack([height, width, 3]))
			label = tf.reshape(label, tf.stack([height,width]))
			image = tf.cast(image,DAT_TYPE)
			return (image,label)

		self._img_splits = {}
		self._truth_splits = {}
		for split,splitname in zip([pe.TRAIN,pe.TEST,pe.VAL],['Train','Test','Val']):
			tfrecord = tf.data.TFRecordDataset(os.path.join(dataset_dir,'TFRecords','%s.tfrecord' % splitname))
			dataset = tfrecord.map(_parse_function)
			if split == pe.TRAIN:
				dataset = dataset.shuffle(buffer_size=1000)
			if split == pe.TEST:
				dataset = dataset.batch(1)
			else:
				dataset = dataset.batch(batch_size)
				dataset = dataset.repeat()
			iterator = dataset.make_one_shot_iterator()
			self._img_splits[split], self._truth_splits[split] = iterator.get_next()
			
		self._img = tf.case({tf.equal(self._partition, pe.TRAIN): lambda: self._img_splits[pe.TRAIN],
							tf.equal(self._partition, pe.TEST): lambda: self._img_splits[pe.TEST],
							tf.equal(self._partition, pe.VAL): lambda: self._img_splits[pe.VAL]})
		self._truth = tf.case({tf.equal(self._partition, pe.TRAIN): lambda: self._truth_splits[pe.TRAIN],
							tf.equal(self._partition, pe.TEST): lambda: self._truth_splits[pe.TEST],
							tf.equal(self._partition, pe.VAL): lambda: self._truth_splits[pe.VAL]})
		#need to re-declare shapes after switch case
		self._img.set_shape([None,None,None,3])
		self._truth.set_shape([None,None,None])
		
	def _return_iterator(self,partition,batch_size):
		"Helper for setting up iterators and batch"
				
	def inputs(self):
		return self._img
	
	def seg_target(self):
		return self._truth
	
	#function to call before actually training batch
	def feed_dict(self,partition,batch_size):
		if partition == pe.TEST:
			assert batch_size == 1
		else:
			assert batch_size == self._batch_size
		return {self._partition:partition}
		
	def base_fcn_weight_dir(self):
		return self._base_fcn_weight_dir
			
	def num_labels(self):
		return self._num_labels
		
	def num_data_items(self,partition):
		if partition == pe.TRAIN:
			return self._num_train
		if partition == pe.TEST:
			return self._num_test
		if partition == pe.VAL:
			return self._num_val
			
	#we have not implemented methods for remapping here
	def semantic_presoftmax(self,ind,partition):
		raise NotImplementedError
			
	def semantic_prob(self,ind,partition):
		raise NotImplementedError
					
	def map_mat(self):
		raise NotImplementedError
		
