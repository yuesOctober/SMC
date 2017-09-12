import csv
import xlrd
import re
import numpy as np
# p=re.compile('^tlfb_m\w\w_d\w\w_4$')
# q=re.compile('^tlfb_m\w\w_d\w\w_3$')

# excelname1="Exp_data_1.xlsx"

# workbook1 = xlrd.open_workbook(excelname1)

# sheet1 = workbook1.sheet_by_index(0)

# indexes=[]
# cols0=sheet1.row_values(0)
# for i in range(len(cols0)):
# 	if p.match(cols0[i]):
# 		indexes.append(i)



# for row in range(1,sheet1.nrows):
# 	val=0
# 	tol=0
# 	for col in range(sheet1.ncols):

# 		if col in indexes:

# 			tol += 1
# 			if sheet1.cell(row,col).value:
				
# 				val += int(sheet1.cell(row,col).value)
# 	print("row %d,  total number is: %d, non zero value is: %d." % (row,tol,val))


def readFile(filename):
	p=re.compile('^tlfb_m\w\w_d\w\w_4$')


	#xsexcelname1="Exp_data_1.xlsx"
	excelname1=filename

	workbook1 = xlrd.open_workbook(excelname1)

	sheet1 = workbook1.sheet_by_index(0)

	indexes=[]
	cols0=sheet1.row_values(0)
	for i in range(len(cols0)):
		if p.match(cols0[i]):
			indexes.append(i)


	A=[]
	for row in range(1,2):
		for col in range(sheet1.ncols):
			if col in indexes:
				if sheet1.cell(row,col).value:	
					A.append(int(sheet1.cell(row,col).value))
				else:
					A.append(0)
	#print A
	A=np.array(A)
	q=re.compile('^tlfb_m\w\w_d\w\w_3$')
	indexes=[]
	cols0=sheet1.row_values(0)
	for i in range(len(cols0)):
		if q.match(cols0[i]):
			indexes.append(i)


	S=[]
	for row in range(1,2):
		for col in range(sheet1.ncols):
			if col in indexes:
				if sheet1.cell(row,col).value:	
					S.append(int(sheet1.cell(row,col).value))
				else:
					S.append(0)
	#print S

	S=np.array(S)
	return S,A

# S,A=readFile("Exp_data_1.xlsx")
# print S
# print A
# def calMLE(S,A):
# 	tols0a0s0=0.0
# 	tols0a0s1=0.0
# 	tols0a1s0=0.0
# 	tols0a1s1=0.0
# 	tols1a0s0=0.0
# 	tols1a0s1=0.0
# 	tols1a1s0=0.0
# 	tols1a1s1=0.0
# 	probs0_s0a0=0.0
# 	probs1_s0a0=0.0
# 	probs0_s0a1=0.0
# 	probs1_s0a1=0.0
# 	probs0_s1a0=0.0
# 	probs1_s1a0=0.0
# 	probs0_s1a1=0.0
# 	probs1_s1a1=0.0
# 	total=len(S)
# 	for i in range(len(S)-1):
# 		if S[i]==0 and A[i]==0 and S[i+1]==0:
# 			#print i
# 			tols0a0s0 += 1
# 		elif S[i]==0 and A[i]==0 and S[i+1]==1:
# 			tols0a0s1 += 1
# 		elif S[i]==0 and A[i]==1 and S[i+1]==0:
# 			tols0a1s0 += 1
# 		elif S[i]==0 and A[i]==1 and S[i+1]==1:
# 			tols0a1s1 += 1
# 		elif S[i]==1 and A[i]==0 and S[i+1]==0:
# 			tols1a0s0 += 1
# 		elif S[i]==1 and A[i]==0 and S[i+1]==1:
# 			#print i
# 			tols1a0s1 += 1
# 		elif S[i]==1 and A[i]==1 and S[i+1]==0:
# 			tols1a1s0 += 1
# 		elif S[i]==1 and A[i]==1 and S[i+1]==1:
			
# 			tols1a1s1 += 1
# 	if tols0a0s0+tols0a0s1:
# 		probs0_s0a0=tols0a0s0/(tols0a0s0+tols0a0s1)
# 		probs1_s0a0=tols0a0s1/(tols0a0s0+tols0a0s1)
# 	if tols0a1s0+tols0a1s1:
# 		probs0_s0a1=tols0a0s0/(tols0a1s0+tols0a1s1)
# 		probs1_s0a1=tols0a0s1/(tols0a1s0+tols0a1s1)
# 	if tols1a0s0+tols1a0s1:
# 		probs0_s1a0=tols1a0s0/(tols1a0s0+tols1a0s1)
# 		probs1_s1a0=tols1a0s1/(tols1a0s0+tols1a0s1)
# 	if tols1a1s0+tols1a1s1:
# 		probs0_s1a1=tols1a1s0/(tols1a1s0+tols1a1s1)
# 		probs1_s1a1=tols1a1s1/(tols1a1s0+tols1a1s1)
# 	Prob=np.array([probs0_s0a0,probs1_s0a0,probs0_s0a1,probs1_s0a1,probs0_s1a0,probs1_s1a0,probs0_s1a1,probs1_s1a1])
# 	total=np.array([tols0a0s0,tols0a0s1,tols0a1s0,tols0a1s1,tols1a0s0,tols1a0s1,tols1a1s0,tols1a1s1])
# 	return total
# total=calMLE(S,A)
# print len(S)
# print total



