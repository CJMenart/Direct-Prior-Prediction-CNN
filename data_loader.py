"Interface for the class that serves as the mediator between net program and the outside world, included numerous pieces of large data it must load to run."
from abc import ABC, abstractmethod


class IDataLoader(ABC):

	@abstractmethod
	def base_fcn_weight_dir(self):
		raise NotImplementedError
		
	@abstractmethod
	def test_img_and_truth(self,ind):
		raise NotImplementedError
	
	@abstractmethod
	def train_img_and_truth(self,ind):
		raise NotImplementedError
	
	@abstractmethod
	def val_img_and_truth(self,ind):
		raise NotImplementedError
	
	@abstractmethod
	def num_labels(self):
		raise NotImplementedError
	
	@abstractmethod
	def num_test(self):
		raise NotImplementedError
	
	@abstractmethod
	def num_train(self):
		raise NotImplementedError
		
	def num_val(self):
		raise NotImplementedError
	
	#here and below methods are for remapping stuff. Leave them un-implemented if you don't want to deal with that.
	#TODO: Actually there are probably more methods for remapping.
	@abstractmethod
	def train_semantic_presoftmax(self,ind):
		raise NotImplementedError
		
	@abstractmethod
	def val_semantic_presoftmax(self,ind):
		raise NotImplementedError
	
	@abstractmethod
	def train_semantic_prob(self,ind):
		raise NotImplementedError
		
	@abstractmethod
	def val_semantic_prob(self,ind):
		raise NotImplementedError

	#TODO: do we need a 'sampled remap targets' here? Depends on future
			
	@abstractmethod
	def map_mat(self):
		raise NotImplementedError