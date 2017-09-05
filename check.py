import csv
import xlrd
import re
p=re.compile('^tlfb_m\w\w_d\w\w_4$')


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
	





