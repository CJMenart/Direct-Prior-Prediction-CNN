"encapsulates loading a matric--can be stored as an image, mat-file, or csv. Maybe even pkl? Loading pkls and image files is very easy with cv2 and python, but there's logic in here for easily getting a mat's and csvs. Also includes a function that tries to auto-detect and load anything.\
h5py cannot be installed straight from pip for python 3, go here: https://pypi.org/project/h5py/2.7.0rc2/#description"

import scipy.io as sio
import csv
import cv2
import numpy as np
import h5py

READ_FLOAT = 0
READ_INT = 1
READ_STR = 2

#NOTE TO SELF: Share with CVL once has been tested in rest of code and no additional bugs/todo found
#TODO: Somehow fix this so that there's, like...always a cast to a specific data type, consistently across all filetypes? Not sure if easy.


def read_any_mat(fname):
	"generic function to try and read any type of file as a float matrix. Be careful, makes some assumptions about file and data."
	ext = fname[fname.rfind('.')+1:]
	
	if ext == 'mat':
		#load mat-file and extract first variable we find
		matrix = read_matfile(fname)
	
	elif ext == 'csv' or ext == 'txt':
		matrix = read_csv(fname,READ_FLOAT)
		
	else:
		matrix = cv2.imread(fname)
		if matrix is None:
			raise Exception("Unrecognized file extension.")
			
	return np.array(matrix)
	

def read_csv(filename,datype):
	"filename: filename\
	datpye: csvr.READ_FLOAT, READ_INT, or READ_STR. Data in file must have the specified form.\
	Returns: python native array with data cast to specified type, or list of strings in the case of strings."
	
	f = open(filename, 'r')
	reader = csv.reader(f)
	matrix = []
	for line in reader:
		if datype == READ_FLOAT:
			row = list(map(float,line))
		elif datype == READ_INT:
			row = list(map(int,line))
		elif datype == READ_STR:
			row = line[0]
		else:
			print('Unrecognized datatype.')
			quit()
		matrix.append(row)
	f.close()
	return matrix
	

def read_matfile(fname,key=None):
	"reads a MAT-file. Optional key argument can specify a name for the variable inside the MAT-file--otherwise we will just grab the first key we find. You can also specify a list of keys to be returned as a list of variables.\
	All values read are returned as np arrays. Presently you must cast string values to strings yourself if you know you're dealing with one."
	#load mat-file and extract first variable we find
	#we use sio unless mat-file version is 7.3, then we use h5py.
	#unfortunately, I only know how to detect version of MATLAB by trying to load with sio and catching error
	try:
		mat = sio.loadmat(fname)
		matrix = _extract_by_key_from_matfile(mat,key)
	except NotImplementedError:
		with h5py.File(fname,'r') as file:
			matrix = _extract_by_key_from_matfile(file,key)
			
	return matrix
	
	
def _extract_by_key_from_matfile(mat,key):
	if key:
		if type(key) is list:
			matrix = []
			for k in key:
				matrix.append(_copy_var_from_matfile(mat[k]))
		else:
			matrix = _copy_var_from_matfile(mat[key])
	else:
		keys = mat.keys()
		keys = [key for key in keys if not key[0] == '_']
		matrix = _copy_var_from_matfile(mat[keys[0]])
	return matrix
	
def _copy_var_from_matfile(var):
	var = np.array(var)
	if var.size == 1:
		var = np.squeeze(var) #squeeze scalars to 1d
	return var
		