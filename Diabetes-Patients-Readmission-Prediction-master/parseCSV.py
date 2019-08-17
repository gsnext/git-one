# Author: Xing Yifan Yix14021
import numpy as np
import math
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import csv
import matplotlib.pyplot as plt

def saveSparse (array, filename):
	np.savez(filename,data = array.data ,indices=array.indices,indptr =array.indptr, shape=array.shape )

def loadSparse(filename):
	loader = np.load(filename)
	return csc_matrix((  loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def saveArray (array, filename):
	np.savez(filename,data = array)

def loadArray(filename):
	loader = np.load(filename)
	return np.array(loader['data'])


def main():
	#csv_data = np.genfromtxt ('diabetic_data.csv', delimiter=",")
	N_SAMPLES = 101766
	N_DRUGS = 23
	N_NUMERICAL = 8 # 1 column for age, the others for the numerical values
	N_CATEGORICAL = 3

	# the three lable vectors
	# Readmission label: 3 classes:
	# 1 for No readimission
	# 2 for <30 days
	# 3 for >30 days
	label_readmission = np.zeros(N_SAMPLES,dtype = int)
	readmission_Dict = {
	'NO':1,
	'<30':2,
	'>30':3
	}

	label_readmission_two_class = np.zeros(N_SAMPLES, dtype = int)
	readimission_Two_Class_Dict = {
	'NO':0,
	'<30':1,
	'>30':1
	}

	# HBA1C label: 
	# 4 classes:
	# 1 for >8
	# 2 for >7
	# 3 for Norm (<7)
	# 4 for None
	label_HBA1C = np.zeros(N_SAMPLES,dtype = int)
	HBA1C_Dict = {
	'>8':1,
	'>7':2,
	'Norm':3,
	'None':4
	}
	
	# Primary Diagonosis Label
	# 9 classes:
	#1 for A disease of the circulatory system
	#2 for Diabetes
	#3 for A disease of the respiratory system
	#4 for Diseases of the digestive system
	#5 for Injury and poisoning
	#6 for Diseases of the musculoskeletal system and connective tissue
	#7 for Diseases of the genitourinary system
	#8 for Neoplasms
	#9 for Other
	label_diag1 = np.zeros(N_SAMPLES,dtype = int)

	# Medication change label: binary, 0 for NO, 1 for Yes
	label_medication_change = np.zeros(N_SAMPLES,dtype = int)

	#Categorical Data Array (N_samples, 3)
	data_categorical = np.zeros(shape = (N_SAMPLES, N_CATEGORICAL), dtype= np.float64)

	# Numerical Data Array
	data_numerical = np.zeros(shape = (N_SAMPLES, N_NUMERICAL), dtype= np.float64)
	# Bag of Drugs
	data_bagOfDrugs = np.zeros(shape = (N_SAMPLES, N_DRUGS), dtype = int)
	# Bag of Drug Dict
	bagOfDrugs_Dict ={
	'Down':1,
	'Up':1,
	'Steady':1,
	'No':0
	}
	age_dict = { # 1 is yong, 2 is medain, 3 is median old, 4 is old, 5 is extreme old
	0:1,
	1:1,
	2:1,
	3:1,
	4:1,
	5:2,
	6:3,
	7:4,
	8:5,
	9:5
	}

	missingcount = 0
	E_count = 0
	V_count = 0
	with open('diabetic_data.csv') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		i = 0
		# skip the first iteration
		iterrows = iter(reader)
		next(iterrows)
		for row in iterrows:
			data_categorical[i][0:N_CATEGORICAL] = [int(num) for num in row[6:6+N_CATEGORICAL]]

			#parse Age, age is in column 4
			age = int(row[4][1:row[4].find('-')])
			data_numerical[i][0] = age_dict[age/10] # age class is depending on the histogram distribution
			#parse Time in the hospital, which is column 9
			data_numerical[i][1] = int (row[9])
			#parse the rest of the numerical data, which is in column 12:18
			data_numerical[i][2:] = [int(num) for num in row [12:18]]

			#parse bag of drugs, which is in column 24:47
			drug_dosage = row[24:47]
			for j in range (0, len(drug_dosage)):
				data_bagOfDrugs[i][j] = bagOfDrugs_Dict[drug_dosage[j]]
			
			#parse the medication change label, which is the 3rd last column
			if(row[-3] == 'No'):
				label_medication_change[i]=0
			else:
				label_medication_change[i]=1

			#parse the readimission label, which is last column
			label_readmission[i] = readmission_Dict[row[-1]]
			label_readmission_two_class[i] = readimission_Two_Class_Dict[row[-1]]

			#parse the HBA1C test label, which is column 23
			label_HBA1C[i] = HBA1C_Dict[row[23]]
			#parse the Primary Diagonosis, which is column 18
			print "The Diag1 Raw:", row[18]
			if(row[18].find('?') != -1):
				label_diag1[i] = 9
				missingcount +=1
			elif(row[18].find('V') !=-1):
				label_diag1[i] = 9
				V_count += 1
			elif(row[18].find('E')!=-1):
				label_diag1[i] = 9
				E_count += 1
			else:

				diag1 = int(float(row[18]))

				print diag1
				if(( diag1>= 390 and diag1<=459) or diag1 == 785):
					label_diag1[i] = 1
				elif(diag1 == 250):
					label_diag1[i] = 2
				elif((diag1 >= 468 and diag1 <=519) or diag1 == 786):
					label_diag1[i] = 3
				elif((diag1 >= 520 and diag1 <=579) or diag1 == 787):
					label_diag1[i] = 4
				elif(diag1 >= 800 and diag1 <=999):
					label_diag1[i] = 5
				elif(diag1 >= 710 and diag1 <=739):
					label_diag1[i] = 6
				elif((diag1 >= 580 and diag1 <=629) or diag1 == 788):
					label_diag1[i] = 7
				elif(diag1 >= 140 and diag1 <=239):
					label_diag1[i] = 8
				else:
					label_diag1[i] = 9

			i+= 1



	# print np.unique(data_numerical[:,0])
	# numBins = 10
	# plt.xlabel("Age class")
	# plt.ylabel("frequency")
	# hist = plt.hist(data_numerical[:,0],numBins,color='green',alpha=0.8)
	# plt.show(hist)

	print "data Categorical first row:", data_categorical[0]
	print "data Categorical shape:",data_categorical.shape
	

	print "data_numerical:\n",data_numerical[-1:,] # see the last patient for verification
	print "data_bagOfDrugs:\n",data_bagOfDrugs
	print "label_readmission:\n",label_readmission
	print "label_readmission_two_class:\n", label_readmission_two_class
	print "label_medication_change:\n", label_medication_change
	print "label_HBA1C:\n", label_HBA1C
	print "label_diag1:\n", label_diag1
	
	print "\nE_count:", E_count
	print "V_count:", V_count
	print "diagnosis missingcount:", missingcount

	# raise ValueError ("purpose stop")

	#saveArray(data_categorical, "data_categorical")
	#saveArray(data_numerical, "data_numerical_age_quantized")
	#saveArray(data_numerical, "data_numerical")
	#saveSparse(csc_matrix(data_bagOfDrugs), "data_bagOfDrugs_sparse")
	#saveArray(label_readmission, "label_readmission")
	#saveArray(label_diag1,"label_diag1")
	#saveArray(label_HBA1C, "label_HBA1C")
	#saveArray(label_medication_change, "label_medication_change")

	print "Number of readmissions:", len(label_readmission_two_class[label_readmission_two_class == 1])
	print "Number of patients who did not readmit:", len(label_readmission_two_class[label_readmission_two_class == 0])

	print "Number of drug changes:", len(label_medication_change[label_medication_change==1])
	print "Number of patients who did not readmit:", len(label_medication_change[label_medication_change == 0])



	#saveArray(label_readmission_two_class, "label_readmission_two_class")

	#print np.asarray(label_readmission)
	return
if __name__ == "__main__":
	main()
