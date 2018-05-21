#marshalls parameters to launch siameseImageComparison on HPC
import PriorClassification as net
import sys, os
import numpy as np
import argparse

#only param is size of training set to use
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("checkpointDir")
	parser.add_argument("--regularizationWeight",type=float,default=1e-5)
	parser.add_argument("--dataset",type=str,default="PASCAL_Context")
	parser.add_argument("--optimizerType",default='Adam')
	parser.add_argument("--momentum",type=float,default=0.99)
	parser.add_argument("--useNesterov",type=bool,default=False)
	parser.add_argument("--batchSize",type=int,default=1)
	parser.add_argument("--learnRate",type=float,default=1e-3)
	parser.add_argument("--evalMode",type=bool,default=False)
	parser.add_argument("--maxIter",type=int,default=1000000)
	parser.add_argument("--mobilePaths",type=bool,default=False) #debug option basically
	parser.add_argument("--weightFile",type=str,default=None)
	parser.add_argument("--trainNameDir",type=str,default=None)
	parser.add_argument("--testNameDir",type=str,default=None)
	parser.add_argument("--imDir",type=str,default=None)
	parser.add_argument("--truthDir",type=str,default=None)
	parser.add_argument("--iterEndOnlyTraining",type=int,default=1200)
	parser.add_argument("--clipGradients",type=bool,default=False)
	parser.add_argument("--remappingLossWeight",type=float,default=0)
	parser.add_argument("--presoftmaxDir",type=str,default=None)
	parser.add_argument("--batchesPerValCheck",type=int,default=50)
	parser.add_argument("--fixBN",type=bool,default=False)
	parser.add_argument("--dropProb",type=float,default=0.5)
	parser.add_argument("--widthHidLayers",type=int,default=2048)
	parser.add_argument("--numHidLayers",type=int,default=1)
	parser.add_argument("--isDistribution",type=bool,default=False)
	parser.add_argument("--weightMultiClassLoss",type=bool,default=False)
	parser.add_argument("--mapMat",type=str,default=None)
	parser.add_argument("--base",type=str,default='resnet_v2')
	
	args = parser.parse_args()
	
	paths = {}
	netOpts = {}

	if args.mobilePaths:
		paths['checkpointDir'] = args.checkpointDir
		paths['trainNameDir'] = args.trainNameDir
		paths['weightFile'] = args.weightFile
		paths['testNameDir'] = args.testNameDir
		paths['imDir'] = args.imDir
		paths['truthDir'] = args.truthDir
		paths['presoftmaxDir'] = args.presoftmaxDir
		paths['mapMat'] = args.mapMat
	else:
		baseDir = '/p/work1/workspace/cmenart/'
		paths['checkpointDir'] = os.path.join(baseDir,args.dataset,args.checkpointDir)
		paths['testNameDir'] = os.path.join(baseDir,'Prior Classification ' + args.dataset,'Testing Data')
		if args.weightFile:
			paths['weightFile'] = os.path.join(baseDir,'Prior Classification ' + args.dataset,args.weightFile) 
		else:
			paths['weightFile'] = None
		paths['trainNameDir'] = os.path.join(baseDir,'Prior Classification ' + args.dataset,'Training Data')
		paths['imDir'] = os.path.join(baseDir, args.dataset, 'Images/')
		paths['truthDir'] = os.path.join(baseDir,args.dataset,'Ground Truth CSV/')
		paths['presoftmaxDir'] = os.path.join(baseDir,args.dataset,'Presoftmax CSV/')
		paths['trainClustering'] = os.path.join(paths['trainNameDir'],'train_clustering_0.csv')
		paths['valClustering'] = os.path.join(paths['trainNameDir'],'val_clustering_0.csv')
		paths['mapMat'] = 'TODO'
		
	netOpts['batchesPerValCheck'] = args.batchesPerValCheck
	netOpts['regularizationWeight'] = args.regularizationWeight
	netOpts['optimizerType'] = args.optimizerType
	netOpts['momentum'] = args.momentum
	netOpts['batchSize'] = args.batchSize
	netOpts['epsilon'] = 1e-12
	netOpts['adamEpsilon'] = 1e-4
	netOpts['learnRate'] = args.learnRate
	netOpts['maxIter'] = args.maxIter
	netOpts['useNesterov'] = args.useNesterov
	netOpts['iterEndOnlyTraining'] = args.iterEndOnlyTraining
	netOpts['clippedGradients'] = args.clipGradients
	netOpts['remappingLossWeight'] = args.remappingLossWeight
	netOpts['fixBN'] = args.fixBN
	netOpts['dropProb'] = args.dropProb
	netOpts['widthHidLayers'] = args.widthHidLayers
	netOpts['numHidLayers'] = args.numHidLayers
	netOpts['isDistribution'] = args.isDistribution
	netOpts['weightMultiClassLoss'] = args.weightMultiClassLoss
	netOpts['base'] = args.base
	
	#currently unused, may implement later
	netOpts['isImSizeFixed'] = False
	
	if args.dataset == 'MS_COCO':
		netOpts['numLabels'] = 90
	elif args.dataset == 'PASCAL_Context':
		netOpts['numLabels'] = 59
	elif args.dataset == 'ADE20K':
		netOpts['numLabels'] = 150
	elif args.dataset == 'NYUDv2':
		netOpts['numLabels'] = 40
	else:
		print('NetTest: Error: unrecognized dataset name')
		
	if args.evalMode:
		net.testing(paths,netOpts)
	else:
		net.train_on_all(paths,netOpts)