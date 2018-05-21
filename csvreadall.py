import csv

READ_FLOAT = 0
READ_INT = 1
READ_STR = 2
	
def readall(filename,datype):
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
			print('Error, unrecognized datype.')
			quit()
		results.append(row)
	f.close()
	return results