"This class encapsulates logic for generating loss and score metrics by remapping base-classifier probabilities to final probabilities using a class histogram and remapping matrix. This performs these operations in tensorflow"

import tensorflow as tf
import numpy as np
from csvreadall import *
import loss_functions
import os

#hist is batch x classes, base_prob is batch x pix x classes, truth is batch x pixels


#WARNING: batch size cannot be a tensor. Must be fixed to include this in graph
#Also, 'batch_size' doesn't necessarily mean batch size. If you're passing in images
#without resizing or padding, one at a time, then this is one.
def conf_remap_loss_and_metrics(map_mat,hist,base_prob,truth,batch_size,epsilon):
	truth_vec = tf.one_hot(truth,base_prob.shape.as_list()[2]+1,on_value=1.0,off_value=0.0,axis=2)[:,:,1:]
	#print('TruthVec:')
	#print(truthVec.shape.as_list())
	#mapped = map(lambda x,y: remap(map_mat,x,y),(hist,base_prob))
	mapped = []
	mats = []
	for b in range(batch_size):
		re,mat = remap(map_mat,hist[b,:],base_prob[b,:,:],epsilon)
		mapped.append(re)
		mats.append(mat)
	#mapped = tf.map_fn(lambda x,y: remap(map_mat,x,y),(hist,base_prob))
	#mapped = remap(map_mat,hist[0],base_prob[0])
	mapped = tf.stack(mapped,0)
	mats = tf.stack(mats)
	#debug
	print('Mapped:')
	print(mapped.shape.as_list())
	print('base_prob:')
	print(base_prob.shape.as_list())
	print('truth_vec:')
	print(truth_vec.shape.as_list())
	loss = loss_functions.categorical_cross_entropy_loss(mapped,truth_vec,epsilon)
	acc = remap_metrics(mapped,truth)
	
	return (loss,acc,(mapped,mats))
	
def remap(map_mat,hist,base_prob,epsilon):
	#first, construct new mapping matrix
	map_mat = tf.transpose(map_mat*hist)
	#normalize
	nmap_mat = map_mat/(tf.reduce_sum(map_mat,axis=0)+epsilon)
	#then, remap
	#print('map_mat')
	#print(map_mat.shape.as_list())
	remapped = tf.transpose(tf.matmul(nmap_mat,tf.transpose(base_prob)))
	return (remapped,nmap_mat)
	
def remap_metrics(out,truth):
	labels = tf.argmax(out,axis=2)+1
	acc = tf.reduce_sum(tf.cast(tf.equal(labels, truth),tf.float32)) /\
			tf.reduce_sum(tf.cast(tf.not_equal(truth, 0),tf.float32))
	'''
	#TODO: Add IOU
	iou = tf.reduce_mean(tf.reduce_sum()/tf.reduce_sum()  )
	'''
	return acc
