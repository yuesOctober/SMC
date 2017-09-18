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

	An=[]
	
	for row in range(1,2):
		A=[]
		for col in range(sheet1.ncols):
			if col in indexes:
				if sheet1.cell(row,col).value:	
					A.append(int(sheet1.cell(row,col).value))
				else:
					A.append(0)
		An.append(A)

	#print A
	An=np.array(An)
	q=re.compile('^tlfb_m\w\w_d\w\w_3$')
	indexes=[]
	cols0=sheet1.row_values(0)
	for i in range(len(cols0)):
		if q.match(cols0[i]):
			indexes.append(i)

	Sn=[]
	
	for row in range(1,2):
		S=[]
		for col in range(sheet1.ncols):
			if col in indexes:
				if sheet1.cell(row,col).value==1:	

					S.append(int(sheet1.cell(row,col).value))
				elif sheet1.cell(row,col).value:
					S.append(2)
				else:
					S.append(0)
		Sn.append(S)
	#print S

	Sn=np.array(Sn)
	return Sn,An


def readFilen(filename):
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

	An=[]
	
	for row in range(1,sheet1.nrows):
		A=[]
		for col in range(sheet1.ncols):
			if col in indexes:
				if sheet1.cell(row,col).value:	
					A.append(int(sheet1.cell(row,col).value))
				else:
					A.append(0)
		An.append(A)

	#print A
	An=np.array(An)
	q=re.compile('^tlfb_m\w\w_d\w\w_3a$')
	indexes=[]
	cols0=sheet1.row_values(0)
	for i in range(len(cols0)):
		if q.match(cols0[i]):
			indexes.append(i)

	Sn=[]
	
	for row in range(1,sheet1.nrows):
		S=[]
		for col in range(sheet1.ncols):
			if col in indexes:
				if sheet1.cell(row,col).value==1:	

					S.append(int(sheet1.cell(row,col).value))
				elif sheet1.cell(row,col).value:
					S.append(2)
				else:
					S.append(0)
		Sn.append(S)
	#print S

	Sn=np.array(Sn)
	return Sn,An

# print S
# print A
# np.savetxt('States_History.txt', S) 
# np.savetxt('Actions_History.txt', A) 

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
# S,A=readFile("Exp_data_1.xlsx")
# total=calMLE(S,A)
# print len(S)
# print total

def calMLE(S,A):
	tols0a0s0=0.0
	tols0a0s1=0.0
	tols0a1s0=0.0
	tols0a1s1=0.0
	tols1a0s0=0.0
	tols1a0s1=0.0
	tols1a1s0=0.0
	tols1a1s1=0.0
	probs0_s0a0=0.0
	probs1_s0a0=0.0
	probs0_s0a1=0.0
	probs1_s0a1=0.0
	probs0_s1a0=0.0
	probs1_s1a0=0.0
	probs0_s1a1=0.0
	probs1_s1a1=0.0
	total=len(S)
	for i in range(len(S)-1):
		if S[i]==0 and A[i]==0 and S[i+1]==0:
			tols0a0s0 += 1
		elif S[i]==0 and A[i]==0 and S[i+1]==1:
			tols0a0s1 += 1
		elif S[i]==0 and A[i]==1 and S[i+1]==0:
			tols0a1s0 += 1
		elif S[i]==0 and A[i]==1 and S[i+1]==1:
			tols0a1s1 += 1
		elif S[i]==1 and A[i]==0 and S[i+1]==0:
			tols1a0s0 += 1
		elif S[i]==1 and A[i]==0 and S[i+1]==1:
			tols1a0s1 += 1
		elif S[i]==1 and A[i]==1 and S[i+1]==0:
			tols1a1s0 += 1
		else:
			tols1a1s1 += 1
	if tols0a0s0+tols0a0s1:
		probs0_s0a0=tols0a0s0/(tols0a0s0+tols0a0s1)
		probs1_s0a0=tols0a0s1/(tols0a0s0+tols0a0s1)
	if tols0a1s0+tols0a1s1:
		probs0_s0a1=tols0a0s0/(tols0a1s0+tols0a1s1)
		probs1_s0a1=tols0a0s1/(tols0a1s0+tols0a1s1)
	if tols1a0s0+tols1a0s1:
		probs0_s1a0=tols1a0s0/(tols1a0s0+tols1a0s1)
		probs1_s1a0=tols1a0s1/(tols1a0s0+tols1a0s1)
	if tols1a1s0+tols1a1s1:
		probs0_s1a1=tols1a1s0/(tols1a1s0+tols1a1s1)
		probs1_s1a1=tols1a1s1/(tols1a1s0+tols1a1s1)
	#print tols1a0s1
	Prob=np.array([probs0_s0a0,probs1_s0a0,probs0_s0a1,probs1_s0a1,probs0_s1a0,probs1_s1a0,probs0_s1a1,probs1_s1a1])
	return Prob


def calMLEn(S,A):

	nState=len(np.unique(S))
	nAction=len(np.unique(A))
	states=np.array([i for i in range(nState) ])
	actions=np.array([i for i in range(nAction) ])
	total=np.zeros([nState,nAction,nState])
	probability=np.zeros([nState,nAction,nState])
	print states
	print actions
	for i in range(len(S)):
		for j in range(len(S[0])-1):
			for s in states:
				for a in actions:
					for s_next in states:
						if S[i][j]==s and A[i][j]==a and S[i][j+1]==s_next:
							total[s][a][s_next] += 1
							
					
				


	for i in range(nState):
		for j in range(nAction):
			totalij=np.sum(total[i][j])
			for k in range(nState):
				if totalij:
					probability[i][j][k]=total[i][j][k]/totalij
	print total
	print probability

	return probability
#S,A=readFile("Exp_data_20170913.xlsx")
S,A=readFile("Exp_data_1.xlsx")

probability=calMLEn(S,A)
print probability