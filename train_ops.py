"Encapsulates ops we use for training. Contains op to run with data and one post-batch. Logic can be involved if we're sending images one at a time and accumulating gradient."

import tensorflow as tf
import numpy as np
from optimizer_from_string import * 
DEBUG = True

#TODO: I think I can decouple this from other behavior by making it be, like, you pass in any number of collections for it to build optimizers for,
#and then to run ops simply pass in any collections you want to use.

class TrainOpHandler:
	"Will manage tensorflow ops needed for training model, given your settings. After constructing you should run train_op() when passing in new data, and post_batch_op() after every batch."

	def __init__(self,net_opts,loss):

		optimizer = optimizer_from_string(net_opts)
		
		trainable_vars = tf.trainable_variables()	
		self._gradients = optimizer.compute_gradients(loss,var_list=trainable_vars)
		#For debugging: useful if you get a gradient that's 'None'
		if DEBUG:
			print('Gradients:')
			for gv in enumerate(self._gradients):
				print(gv)
			print('End of Gradients.')
		if net_opts['is_clipped_gradients']:
			self._gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in self._gradients] # gradient capping
		self._accum_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainable_vars]
		self._accumulate = [self._accum_gradients[i].assign_add(gv[0]) for i, gv in enumerate(self._gradients)]
		self._apply_grad = optimizer.apply_gradients([(self._accum_gradients[i], gv[1]) for i, gv in enumerate(self._gradients)])
		self._clear_gradients = [tv.assign(tf.zeros_like(tv)) for tv in self._accum_gradients]
		
		if net_opts['iter_end_only_training'] > 0:
			trainable_vars_fresh = tf.get_collection('fresh')
			if DEBUG:
				print(trainable_vars_fresh)
			self._gradients_fresh = optimizer.compute_gradients(loss,var_list=trainable_vars_fresh)
			if net_opts['is_clipped_gradients']:
				self._gradients_fresh = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in self._gradients_fresh] # gradient capping
			self._accum_gradients_fresh = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainable_vars_fresh]
			self._accumulate_fresh = [self._accum_gradients_fresh[i].assign_add(gv[0]) for i, gv in enumerate(self._gradients_fresh)]
			self._apply_grad_fresh = optimizer.apply_gradients([(self._accum_gradients_fresh[i], gv[1]) for i, gv in enumerate(self._gradients_fresh)])
			self._clear_gradients_fresh = [tv.assign(tf.zeros_like(tv)) for tv in self._accum_gradients_fresh]
		
		#possibly could get all update ops that were added to same collection you get variables from?
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		if DEBUG:
			print('update-ops:')
			print([op.name for op in update_ops])
		all_updates = tf.group(*update_ops)
		if DEBUG:
			print('batch-norm updates:')
			print(update_ops)
		self._accumulate = tf.group(*self._accumulate, all_updates)
		
		fresh_updates = tf.group(*[op for op in update_ops if 'fcbn' in op.name])
		self._accumulate_fresh = tf.group(*self._accumulate_fresh, fresh_updates)
		
		self._iter_end_only_training = net_opts['iter_end_only_training']
		
	def train_op(self,iter):
		if iter >= self._iter_end_only_training:
			return self._accumulate
		else:
			return self._accumulate_fresh
	
	def post_batch_actions(self,iter,sess):
		if iter >= self._iter_end_only_training:
			sess.run(self._apply_grad)
			sess.run(self._clear_gradients)
		else:
			sess.run(self._apply_grad_fresh)
			sess.run(self._clear_gradients_fresh)
		
	def check_gradients(self,iter,sess):
		for g in self._accum_gradients[-2:] if iter >= self._iter_end_only_training else self._accum_gradients_fresh[-2:]:
			grad = sess.run(g,feed_dict={})
			if np.isnan(np.sum(grad)):
				raise Exception('Model diverged with nan gradient')
		
		if DEBUG:

			if iter < self._iter_end_only_training:		
				for g in (self._accum_gradients_fresh[-2:]):
					grad = sess.run(g,feed_dict={})
					print('gradients fresh:')
					print(grad)
				for g in (self._accum_gradients[-2:]):
					grad = sess.run(g,feed_dict={})
					print('gradients:')
					print(grad)
			else:
				for g in (self._accum_gradients[-10:]):
					grad = sess.run(g,feed_dict={})
					print('gradients:',text_log)
					print(grad,text_log)
				