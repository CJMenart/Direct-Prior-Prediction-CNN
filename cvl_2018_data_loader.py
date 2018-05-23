"Loads from datasets saved as matfiles, format created for CVL 5/2018.\
This setup basically incompatible with TFRecords. But we're not in production code here, this is quickly-changing research and I am DONE with tfrecord's inflexibility atm."
from data_loader import *
from my_read_mat import *
import scipy.io as sio
import os

#TODO if we want to get fancy we could also scan the actual filesize of the folder
MAX_VAL_PRELOAD = 250
DEBUG = False


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
		
	def base_fcn_weight_dir(self):
		return self._base_fcn_weight_dir

	def test_img_and_truth(self,ind):
		assert ind <= self._num_test
		return self._img_and_truth(ind,'TestImgs','test')
	
	def train_img_and_truth(self,ind):
		assert ind <= self._num_train
		return self._img_and_truth(ind,'TrainImgs','train')
		
	def val_img_and_truth(self,ind):
		assert ind <= self._num_val
		if self._preloaded_val:
			return (self._val_imgs[ind],self._val_truth[ind])
		else:
			return self._img_and_truth(ind,'ValImgs','val')
	
	def _img_and_truth(self,ind,subfolder,prefix):
		img_fname = os.path.join(self._dataset_dir,subfolder,'%s_%06d_img.png' % (prefix,ind+1))
		img = cv2.imread(img_fname)
		truth_fname = os.path.join(self._dataset_dir,subfolder,'%s_%06d_pixeltruth.mat' % (prefix,ind+1))
		truth = read_matfile(truth_fname,'truth_img')
		if DEBUG:
			print('truth')
			print(truth)
		return (img,truth)
	
	def num_labels(self):
		return self._num_labels
	
	def num_test(self):
		return self._num_test
	
	def num_train(self):
		return self._num_train
	
	def num_val(self):
		return self._num_val
		
	#we have not implemented methods for remapping here
	def train_semantic_presoftmax(self,ind):
		raise NotImplementedError
		
	def val_semantic_presoftmax(self,ind):
		raise NotImplementedError
	
	def train_semantic_prob(self,ind):
		raise NotImplementedError
		
	def val_semantic_prob(self,ind):
		raise NotImplementedError
			
	def map_mat(self):
		raise NotImplementedError