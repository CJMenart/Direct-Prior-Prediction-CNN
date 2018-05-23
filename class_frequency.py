"compute histogram of how often object classes occur in training data."

import numpy as np


def class_frequency(data_loader):
	class_freq = np.zeros((1,num_labels),tf.float32)
	for t in range(data_loader.num_train()):
		_,truth = data_loader.train_img_and_truth(t)
		class_freq[np.unique(truth)] += 1
	class_freq = class_freq/len(train_img_names)
	return class_freq