import csv
import xlrd
import re
import numpy as np
p=re.compile('^tlfb_m\w\w_d\w\w_4$')
q=re.compile('^tlfb_m\w\w_d\w\w_3$')

excelname1="Exp_data_1.xlsx"

workbook1 = xlrd.open_workbook(excelname1)

sheet1 = workbook1.sheet_by_index(0)

indexes=[]
cols0=sheet1.row_values(0)
for i in range(len(cols0)):
	if p.match(cols0[i]):
		indexes.append(i)



for row in range(1,sheet1.nrows):
	val=0
	tol=0
	for col in range(sheet1.ncols):

		if col in indexes:

			tol += 1
			if sheet1.cell(row,col).value:
				
				val += int(sheet1.cell(row,col).value)
	print("row %d,  total number is: %d, non zero value is: %d." % (row,tol,val))


# def readFile(filename):
# 	p=re.compile('^tlfb_m\w\w_d\w\w_4$')


# 	#xsexcelname1="Exp_data_1.xlsx"
# 	excelname1=filename

# 	workbook1 = xlrd.open_workbook(excelname1)

# 	sheet1 = workbook1.sheet_by_index(0)

# 	indexes=[]
# 	cols0=sheet1.row_values(0)
# 	for i in range(len(cols0)):
# 		if p.match(cols0[i]):
# 			indexes.append(i)


# 	A=[]
# 	for row in range(1,2):
# 		for col in range(sheet1.ncols):
# 			if col in indexes:
# 				if sheet1.cell(row,col).value:	
# 					A.append(int(sheet1.cell(row,col).value))
# 				else:
# 					A.append(0)
# 	#print A
# 	A=np.array(A)
# 	q=re.compile('^tlfb_m\w\w_d\w\w_3$')
# 	indexes=[]
# 	cols0=sheet1.row_values(0)
# 	for i in range(len(cols0)):
# 		if q.match(cols0[i]):
# 			indexes.append(i)


# 	S=[]
# 	for row in range(1,2):
# 		for col in range(sheet1.ncols):
# 			if col in indexes:
# 				if sheet1.cell(row,col).value:	
# 					S.append(int(sheet1.cell(row,col).value))
# 				else:
# 					S.append(0)
# 	#print S

# 	S=np.array(S)
# 	return S,A

# S,A=readFile("Exp_data_1.xlsx")

# print S,A