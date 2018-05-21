#WARNING: Not yet tested/debugged, donut run

#import time
import numpy as np
import tensorflow as tf
import os
import random
import sys
import cv2
import csvreadall as csvr
import fnmatch
import PriorNetWrapper as net
from augment_img import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import shutil, errno
DEBUG = True
# Chris Menart, 1-9-18
#went back to editing 5-18-18

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
		

#train a base network on the entire dataset, then train it on sub-clusters
#TODO out of date
def train_on_clusters(paths,netOpts):
	global_dir = os.path.join(paths['checkpointDir'],'global')
	training(paths,netOpts,global_dir,netOpts['maxIter'])
	clusters = csvr.readall(paths['clusters'],csvr.READ_INT)
	num_clust = np.max(clusters)
	for clust in range(clusters):
		clust_dir = os.path.join(paths['checkpointDir'],'clust_%d' % clust)
		if not os.path.exists(clust_dir):
			copyanything(global_dir,clust_dir)
		training(paths,netOpts,clust_dir,netOpts['maxIter']*2)

def train_on_all(paths,netOpts):
	trainImgNames = csvr.readall(os.path.join(paths['trainNameDir'], "prior_train_img_names_0.csv"),csvr.READ_STR)
	valImgNames = csvr.readall(os.path.join(paths['trainNameDir'], "prior_val_img_names_0.csv"),csvr.READ_STR)

	training(paths,netOpts,paths['checkpointDir'],netOpts['maxIter'],trainImgNames,valImgNames)	
	
def training(paths,netOpts,checkpointDir,maxIter,trainImgNames,valImgNames):
	#paths, netOpts are both dictionaries containing various options for how we run the network
	#trains one neural network
	
	#Settings and paths and stuff			
	textLog = os.path.join(paths['checkpointDir'], "NetworkLog.txt")
	trainErrLog = os.path.join(paths['checkpointDir'], "TrainErr.csv")
	valErrLog = os.path.join(paths['checkpointDir'], "ValErr.csv")
	doublePrint("Welcome to prior network training.",textLog)	
	if netOpts['remappingLossWeight'] > 0:
		map_mat = np.array(csvr.readall(paths['mapMat']),csvr.READ_FLOAT)
	
	doublePrint("Loading validation images...",textLog)

	val_imgs = []
	val_targets = []
	val_remap_targets = []
	val_base_probs = []
	for v in range(len(valImgNames)):
		img = cv2.imread(os.path.join(paths['imDir'], valImgNames[v]))
		sz = img.shape
		val_imgs.append(img[np.newaxis,:,:,:])
		
		truth = np.array(csvr.readall(os.path.join(paths['truthDir'], valImgNames[v][:-3] + 'csv'),csvr.READ_INT))
		val_targets.append(truth[np.newaxis,:,:])
		
		if netOpts['remappingLossWeight'] > 0:
			remap_samples = np.array(csvr.readall(os.path.join(paths['remapDir'],valImgNames[v][:-3] + "csv"),csvr.READ_FLOAT))
			target = remap_samples[:,0]
			base_probs = remap_samples[:,1:]
			val_remap_targets.append(target)
			val_base_probs.append(base_probs)
			
	#more debugging
	if DEBUG:
		doublePrint('Example validation item',textLog)
		doublePrint(val_imgs[0],textLog)
		doublePrint(val_targets[0],textLog)
	
	#compute average of classes present for binary loss weights
	if not netOpts['isDistribution'] and netOpts['weightMultiClassLoss']:
		class_freq = np.zeros((1,netOpts['numLabels']),tf.float32)
		doublePrint('Computing class frequencies...',textLog)
		for t in range(len(trainImgNames)):
			truth = np.array(csvr.readall(os.path.join(paths['truthDir'],trainImgNames[tInd][:-3] + 'csv'),csvr.READ_INT)).astype('uint16')
			class_freq[np.unique(truth)] += 1
		class_freq = class_freq/len(trainImgNames)
		doublePrint(class_freq,textLog)
	else:
		class_freq = np.ones((1,netOpts['numLabels']),dtype=np.float32) #doesn't matter
	
	tf.reset_default_graph()
	
	#debug to force CPU mode!
	config = tf.ConfigProto(
		device_count = {'GPU': 0}
	)
	sess = tf.InteractiveSession(config=config)
	
	#sess = tf.InteractiveSession()
	network = net.PriorNetWrapper(netOpts)
	bestValLoss = tf.Variable(sys.float_info.max,trainable=False,name="bestValLoss")
	reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	totalLoss = network.loss + reg_loss
	#debugging
	#doublePrint('Regularization Losses:',textLog)
	#doublePrint(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),textLog)
	
	optimizerType = netOpts['optimizerType']
	if optimizerType == 'Adam':
		optimizer = tf.train.AdamOptimizer(netOpts['learnRate'],epsilon=0.01)
	elif optimizerType == 'Momentum':
		optimizer = tf.train.MomentumOptimizer(netOpts['learnRate'],
						netOpts['momentum'],
						use_nesterov=netOpts['useNesterov'])
	elif optimizerType == 'Yellowfin':
		print('error, no yellowfin')
		#optimizer = yellowfin.YFOptimizer(learning_rate=1e-3)
	else:
		print('Error. Unrecognized optimizer type.')
		return	
		
	#TODO: Add option for normal large-batching b/c I don't think it matters here (though will require padding BS...) maybe factor some of this out into new function? Only potential issue is if padding-to-standardize messes with batch norm, and that will require investigation.
	trainableVars = tf.trainable_variables()	
	gradients = optimizer.compute_gradients(totalLoss,var_list=trainableVars)
	#For debugging: useful is you get a gradient that's 'None'
	if DEBUG:
		print('Gradients:')
		for gv in enumerate(gradients):
			print(gv)
		print('End of Gradients.')
	if netOpts['clippedGradients']:
		gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients] # gradient capping
	accumGradients = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainableVars]
	clearGradients = [tv.assign(tf.zeros_like(tv)) for tv in accumGradients]
	accumulate = [accumGradients[i].assign_add(gv[0]) for i, gv in enumerate(gradients)]
	applyGrad = optimizer.apply_gradients([(accumGradients[i], gv[1]) for i, gv in enumerate(gradients)])
	
	
	if netOpts['iterEndOnlyTraining'] > 0:
		trainableVarsF = tf.get_collection('fresh')
		print(trainableVarsF)
		gradientsF = optimizer.compute_gradients(totalLoss,var_list=trainableVarsF)
		if netOpts['clippedGradients']:
			gradientsF = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsF] # gradient capping
		accumGradientsF = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainableVarsF]
		clearGradientsF = [tv.assign(tf.zeros_like(tv)) for tv in accumGradientsF]
		accumulateF = [accumGradientsF[i].assign_add(gv[0]) for i, gv in enumerate(gradientsF)]
		applyGradF = optimizer.apply_gradients([(accumGradientsF[i], gv[1]) for i, gv in enumerate(gradientsF)])
	
	#WARNING: Batch norm ops only updated on end-to-end training.
	#TODO: If you add option for later batch norm, must carefully collect proper update ops here.
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	update_ops = tf.group(*update_ops)
	doublePrint('batch-norm updates:',textLog)
	doublePrint(update_ops,textLog)
	accumulate = tf.group(*accumulate, update_ops)
	
	#tensorboard inspection
	summaries = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(paths['checkpointDir'],graph=sess.graph)
	
	saver = tf.train.Saver()
	latestModel = tf.train.latest_checkpoint(paths['checkpointDir'])
	if not latestModel: #attempt to find model file directly
		filenames = os.listdir(paths['checkpointDir'])
		matches = fnmatch.filter(filenames,"*.meta*")
		if len(matches) > 0:
			latestModel = matches[-1]
		if latestModel:
			latestModel = os.path.join(paths['checkpointDir'],latestModel[:-5])
	if latestModel:
		tf.global_variables_initializer().run()
		saver.restore(sess,latestModel)
		start = int(latestModel.split('-')[-1])+1
		doublePrint('Starting from iteration %d. Current validation loss: %.5f' % (start, np.mean(sess.run([bestValLoss]))),textLog)
	else:
		doublePrint('Creating a new network...',textLog)
		start = 0
		tf.global_variables_initializer().run()
		network.loadWeights(paths['weightFile'],sess,netOpts)
	modelName = os.path.join(paths['checkpointDir'],'ConvPriorNet')
		
	#debugging
	print("Trainable Variables:")
	print(trainableVars)
	
	curTrainIndices = list(range(len(trainImgNames)))
		
	for iter in range(start,netOpts['maxIter']):
				
		trainEndOnly = iter < netOpts['iterEndOnlyTraining']
		if iter == netOpts['iterEndOnlyTraining']:
			doublePrint('Switching to end-to-end training.',textLog)
				
		if iter % (netOpts['batchesPerValCheck']) == 0:
			valLoss = 0
			valErr = 0
			valAcc = 0
			for vInd in range(len(valImgNames)):
				feedDict = {network.inputs:val_imgs[vInd],
							network.seg_target:val_targets[vInd],
							network.class_frequency: class_freq,
							network.isTrain: False}
							
				if netOpts['remappingLossWeight'] > 0:
					feedDict[network.remap_target] = val_remap_targets[vInd]
					feedDict[network.remap_base_prob] = val_base_probs[vInd]
					feedDict[network.map_mat] = map_mat
					
				loss, err, acc, smry = sess.run([network.loss, network.prior_err, network.seg_acc, summaries], feed_dict=feedDict)
				valLoss += loss
				valErr += err
				valAcc += acc
			valLoss = np.mean(valLoss)/len(valImgNames)
			valErr = np.mean(valErr)/len(valImgNames)
			valAcc = np.mean(valAcc)/len(valImgNames)
			doublePrint('step %d: val loss %.5f' % (iter, valLoss),textLog)
			doublePrint('step %d: val error ~= %.5f' % (iter, valErr),textLog)
			doublePrint('step %d: val acc ~= %.5f' % (iter,valAcc),textLog)
			errPrint(valErr,valErrLog)
			summary_writer.add_summary(smry, iter)
			newBest = False
			if valLoss < np.mean(sess.run([bestValLoss])):
				sess.run([bestValLoss.assign(valLoss)])
				doublePrint("NEW VALIDATION BEST",textLog)
				newBest = True
			if newBest or iter % 10000 == 0:
				saver.save(sess, modelName+"Best",global_step = iter)
				
		batchSize = netOpts['batchSize']
			
		loss = 0
		err = 0
		acc = 0
		for item in range(batchSize):
			if len(curTrainIndices) == 0:
				curTrainIndices = list(range(len(trainImgNames)))		
			tInd = curTrainIndices.pop(random.randint(0,len(curTrainIndices)-1))
			
			imName = os.path.join(paths['imDir'], trainImgNames[tInd])
			#print(imName)
			img = cv2.imread(imName)
			sz = img.shape
			truth = np.array(csvr.readall(os.path.join(paths['truthDir'],trainImgNames[tInd][:-3] + 'csv'),csvr.READ_INT)).astype('uint16')
			(img,truth) = augment_img(img,truth)
			#debug
			doublePrint('img,truth:',textLog)
			doublePrint(img,textLog)
			doublePrint(truth,textLog)
			
			feedDict={
				network.inputs:img[np.newaxis,:,:,:],
				network.seg_target: truth[np.newaxis,:,:],
				network.class_frequency: class_freq,
				network.isTrain: True}
				
			if netOpts['remappingLossWeight'] > 0:
				remap_samples = np.array(csvr.readall(os.path.join(paths['presoftmaxDir'],trainImgNames[tInd][:-3] + "csv")),csvr.READ_FLOAT)
				truth = remapSamples[:,0]
				base_probs = remapSamples[:,1:]
				feedDict[network.remap_target] = remap_samples
				feedDict[network.remap_base_prob] = base_probs
				feedDict[network.mapMat] = map_mat
							
			#_, curLoss, curErr, numWindows,scoreMap,procTarg,baseProbs = sess.run([accumulateF if trainEndOnly else accumulate,network.loss,network.err,network.numWindows,network.scoreMap,network.processedTarget,network.baseProbs], feed_dict=feedDict)
			_, curLoss, curErr,curAcc = sess.run([accumulateF if trainEndOnly else accumulate,network.loss,network.prior_err,network.seg_acc], feed_dict=feedDict)
			
			loss += curLoss
			err += curErr
			acc += curAcc
			
		for g in accumGradientsF[-2:] if trainEndOnly else accumGradients[-2:]:
			grad = sess.run(g,feed_dict={})
			if np.isnan(np.sum(grad)):
				doublePrint('Model diverged with nan gradient',textLog)
				quit()
			
		'''
		#debug
		if trainEndOnly:		
			for g in (accumGradientsF[-2:]):
				grad = sess.run(g,feed_dict={})
				doublePrint('gradientsF:',textLog)
				doublePrint(grad,textLog)
			for g in (accumGradients[-2:]):
				grad = sess.run(g,feed_dict={})
				doublePrint('gradients:',textLog)
				doublePrint(grad,textLog)
		else:
			for g in (accumGradients[-10:]):
				grad = sess.run(g,feed_dict={})
				doublePrint('gradients:',textLog)
				doublePrint(grad,textLog)
		'''
		sess.run(applyGradF if trainEndOnly else applyGrad)
		sess.run(clearGradientsF if trainEndOnly else clearGradients)
		
		'''
		for t in trainableVars:
			var = sess.run(t,feed_dict={})
			doublePrint('trainable var:',textLog)
			doublePrint(var,textLog)
		'''
		
		loss = loss/batchSize #avg loss per element, to print
		err = err/batchSize
		acc = acc/batchSize
		doublePrint('step %d: loss %.3f, p-err %.3f, acc = %.3f' % (iter, loss,err,acc),textLog)
		errPrint(err,trainErrLog)
		
		if np.isnan(loss) or np.isnan(err):
			# NOTE: It is theoretically possible,
			# though unlikely, to have a model diverge with NaN loss without a bug
			# if the weights cause an output of the network to overflow to inf
			# plus y'know we want to catch bugs
			doublePrint('Model diverged with loss = %.2f and err = %.2f' % (loss,err),textLog)
			doublePrint('Caused by the following scoremap:',textLog)
			doublePrint(scoreMap,textLog)
			doublePrint('And following procTarg:',textLog)
			doublePrint(procTarg,textLog)
			doublePrint('And following baseProbs:',textLog)
			doublePrint(baseProbs,textLog)
			
			quit()			
		
	sess.close()
	doublePrint("Done.",textLog)

def doublePrint(msg,textLog):
	print(msg)
	try:
		with open(textLog, "a+") as myfile:
			print(msg,file=myfile)
	except Exception:
		pass
		
def errPrint(msg,errLog):
	try:
		with open(errLog, "a+") as myfile:
			print(msg,file=myfile)
	except Exception:
		pass