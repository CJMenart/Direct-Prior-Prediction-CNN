"Interface for the class that serves as the mediator between net program and the outside world, included numerous pieces of large data it must load to run."
from abc import ABC, abstractmethod
#implementation will need import partition_enum


class IDataLoader(ABC):

	@abstractmethod
	def base_fcn_weight_dir(self):
		raise NotImplementedError
		
	@abstractmethod
	def img_and_truth(self,ind,partition):
		raise NotImplementedError
		
	@abstractmethod
	def num_labels(self):
		raise NotImplementedError
	
	@abstractmethod
	def num_data_items(self,partition):
		raise NotImplementedError
		
	#here and below methods are for remapping stuff. Leave them un-implemented if you don't want to deal with that.
	#TODO: Actually there are probably more methods for remapping.
	@abstractmethod
	def semantic_presoftmax(self,ind,partition):
		raise NotImplementedError
			
	@abstractmethod
	def semantic_prob(self,ind,partition):
		raise NotImplementedError
		
	#TODO: do we need a 'sampled remap targets' here? Depends on future
			
	@abstractmethod
	def map_mat(self):
		raise NotImplementedError