"encapsulates the logic for reading all of the data from a CSV file and potentially converting to a numeric type."
import csv

READ_FLOAT = 0
READ_INT = 1
READ_STR = 2

def readall(filename,datype):
	"filename: filename\
	datpye: csvr.READ_FLOAT, READ_INT, or READ_STR. Data in file must have the specified form.\
	Returns: python native array with data cast to specified type, or list of strings in the case of strings."
	
	f = open(filename, 'r')
	reader = csv.reader(f)
	results = []
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
		results.append(row)
	f.close()
	return results