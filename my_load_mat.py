"encapsulates loading a matric--can be stored as an image, mat-file, or csv. We no longer have to care which when calling this function--this loads all correctly.\
It's a little crude. Partially a stopgap. Assumes you are loading ints, that the matfile only has one field, etc."

import scipy.io as sio
import csvreadall as csvr
import cv2
import numpy as np

def my_load_int_mat(fname):
	ext = fname[fname.rfind('.')+1:]
	
	if ext == 'mat':
		#load mat-file and extract first variable we find
		mat = sio.loadmat(fname)
		keys = mat.keys()
		keys = [key for key in keys if not key[0] == '_']
		img = mat[keys[0]]
	
	elif ext == 'csv' or ext == 'txt':
		img = csvr.readall(fname,csvr.READ_INT)
		
	else:
		img = cv2.imread(fname)
		if img is None:
			raise Exception("Unrecognized file extension.")
			
	return np.array(img)