"Interface for the class that serves as the mediator between net program and the outside world, included numerous pieces of large data it must load to run."
from abc import ABC, abstractmethod
#implementation will need import partition_enum


class IDataLoader(ABC):

	#what you actually call when running
	@abstractmethod
	def feed_dict(self,partition,batch_size):
		raise NotImplementedError
		
	@abstractmethod
	def base_fcn_weight_dir(self):
		raise NotImplementedError
	
	@abstractmethod
	def inputs(self):
		raise NotImplementedError
	
	@abstractmethod
	def seg_target():
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
	def semantic_presoftmax(self):
		raise NotImplementedError
			
	@abstractmethod
	def semantic_prob(self):
		raise NotImplementedError
		
	#TODO: do we need a 'sampled remap targets' here? Depends on future
	
	@abstractmethod
	def map_mat(self):
		raise NotImplementedError