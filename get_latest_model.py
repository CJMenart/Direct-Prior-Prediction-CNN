"""Encapsulates some logic for trying a couple different approaches to get any saved model checkpoints out of a folder--could be saved in a couple different ways in TF."""

import tensorflow as tf
import fnmatch
import os


def get_latest_model(checkpoint_dir):

	latest_model = tf.train.latest_checkpoint(checkpoint_dir)
	if not latest_model: #attempt to find model file directly
		filenames = os.listdir(checkpoint_dir)
		matches = fnmatch.filter(filenames,"*.meta*")
		if len(matches) > 0:
			latest_model = matches[-1]
		if latest_model:
			latest_model = os.path.join(checkpoint_dir,latest_model[:-5])