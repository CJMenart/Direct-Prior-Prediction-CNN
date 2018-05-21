import numpy as np
import cv2
import csvreadall as csvr
import os

GAMMA_RANGE = [0.5,2]
AREA_RANGE = [200*200,600*600] 
LB_SCALE_SET = [0.7,0.8,0.9,1.0,1.1,1.2,1.3]

#NOTE: Due to lighting concerns, you are required to call augmentation methods here before any changing of the image range out of the 0-255 space

#augment in the style of LabelBank
def augment_LB(img,truth=None):
	#flipping
	if np.random.randint(0,2) == 0:
		img = np.fliplr(img)
		if truth is not None:
			truth = np.fliplr(truth)
			
	#scaling
	scale = LB_SCALE_SET[np.random.randint(0,len(LB_SCALE_SET))]
	(img,truth) = resizeRatio(img,scale,truth)
	return (img,truth)

def augment_img(img,truth=None,fixedSz=None):

	#flipping
	if np.random.randint(0,2) == 0:
		img = np.fliplr(img)
		if truth is not None:
			truth = np.fliplr(truth)
		
	if fixedSz:
		img = cv2.resize(img, (fixedSz, fixedSz))
	else:
		#randomly resize
		#WARNING: No checks done to ensure length of smallest side
		size = img.shape
		area = size[0]*size[1]
		(img,truth) = resizeRatio(img,np.sqrt(np.random.randint(AREA_RANGE[0],AREA_RANGE[1])/area),truth)
		#scale = LB_SCALE_SET[np.random.randint(0,len(LB_SCALE_SET))]
		#(img,truth) = resizeRatio(img,scale,truth) #commented-out produces mysterious errors but only no HPC :(
		
	#lighting augmentation
	img = gamma_correction(img,np.random.uniform(GAMMA_RANGE[0],GAMMA_RANGE[1]))

	return (img,truth)

def resizeRatio(img,ratio,truth=None):
	size = img.shape
	img = cv2.resize(img, ( int(round(size[1]*ratio)), int(round(size[0]*ratio))))
	if truth is not None:
		truth = cv2.resize(truth,(int(round(size[1]*ratio)), int(round(size[0]*ratio))),interpolation=cv2.INTER_NEAREST)
	return (img,truth)
	
def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, 1/correction)
    return np.uint8(img*255)
	
def test_augment(img_fname,truth_fname,save_dir):
	img = cv2.imread(img_fname)
	truth = np.array(csvr.readall(truth_fname,csvr.READ_INT))
	for trial in range(20):
		aug_img,aug_truth = augment_img(img,truth)
		cv2.imwrite(os.path.join(save_dir,'aug_img%03d.png' % trial),aug_img)
		cv2.imwrite(os.path.join(save_dir,'aug_truth%03d.png' % trial),aug_truth)
	for trial in range(20):
		aug_img,aug_truth = augment_img(img,truth,100)
		cv2.imwrite(os.path.join(save_dir,'aug_img_fixed%03d.png' % trial),aug_img)
		cv2.imwrite(os.path.join(save_dir,'aug_truth_fixed%03d.png' % trial),aug_truth)