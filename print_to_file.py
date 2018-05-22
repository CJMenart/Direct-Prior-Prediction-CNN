"print statement to a text file, opening and closing it within the call. Can print to both file and console at same time."


def double_print(msg,text_log):
	print(msg)
	try:
		with open(text_log, "a+") as myfile:
			print(msg,file=myfile)
	except Exception:
		pass
		
		
def file_print(msg,err_log):
	try:
		with open(err_log, "a+") as myfile:
			print(msg,file=myfile)
	except Exception:
		pass