"put a dataset (assuming mat files with CVL 2018 directory structure) into tfrecords. \
Not called by main program; build records before running. It'll just save it in the dataset \
directory."

import tensorflow as tf
import os
import sys
import cvl_2018_data_loader as loading
import partition_enum as pe
DEBUG = True


def build_tfrecord(dataset_dir):
	print('Building tfrecord from %s...' % dataset_dir)
	net_opts = {'dataset_dir':dataset_dir,'base_fcn_weight_dir':'_'}
	loader = loading.CVL2018DataLoader(net_opts)
	os.makedirs(os.path.join(dataset_dir,'TFRecords'))
	for split,splitname in zip(pe.SPLITS,pe.SPLITNAMES):
		output_file = os.path.join(dataset_dir,'TFRecords','%s.tfrecord' % splitname)
		with tf.python_io.TFRecordWriter(output_file) as record_writer:
			for i in range(loader.num_data_items(split)):
				(img,truth) = loader.img_and_truth(i,split)
				if DEBUG:
					print('truth:')
					print(truth.shape)
				example = tf.train.Example(features=tf.train.Features(
					feature={
						'image': _bytes_feature(img.tostring()),
						'label': _bytes_feature(truth.tostring()),
						'height': _int64_feature(truth.shape[0]),
						'width': _int64_feature(truth.shape[1])
				}))
				record_writer.write(example.SerializeToString())
				print('Wrote im %d in %s' % (i,splitname))
					
	
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
  
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))