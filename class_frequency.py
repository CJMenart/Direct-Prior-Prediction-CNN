"compute histogram of how often object classes occur in training data."

import numpy as np
import csvreadall as csvr

#TODO: Fix data reading
def class_frequency(truth_dir,num_labels,train_img_names):
	class_freq = np.zeros((1,num_labels),tf.float32)
	for t in range(len(train_img_names)):
			truth = np.array(csvr.readall(os.path.join(truth_dir,train_img_names[t][:-3] + 'csv'),csvr.READ_INT)).astype('uint16')
			class_freq[np.unique(truth)] += 1
	class_freq = class_freq/len(train_img_names)
	return class_freq