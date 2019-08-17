# Author: Jim Yifan Xing Yix14021
import numpy as np
import math
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from sklearn import preprocessing

from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import feature_selection as fs
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import collections
import operator

sigma = None

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

def TF_IDF(documentData,filename = "Default_TFIDF"): # document is expected to be dense with shape (N_features, N_samples)
	TF_IDFData = documentData.astype(np.float64)
	idf = np.zeros(shape=(TF_IDFData.shape[0],),dtype=float) # final data matrix
	for i in range(0, TF_IDFData.shape[0]):
		non_empty_row, non_empty_col, values =sp.find(TF_IDFData[i,:])
		if (len(non_empty_col) != 0):
			idf[i] = math.log(float(TF_IDFData.shape[1]) /len(non_empty_col))
		#idf[i] = math.log(float(TF_IDFData.shape[1]) /(1+len(non_empty_col))) # or add 1 to adjust to avoid division by 0
	# print idf
	TF_IDFData = csc_matrix(TF_IDFData)
	if (TF_IDFData.format != 'csc') :
		msg = 'Can only row-normalize in place with csc format'
		msg = msg.format(sps_mat.format)
		raise ValueError(msg)
	# print "indices:", TF_IDFData.indices
	# print np.take(idf, TF_IDFData.indices)
	TF_IDFData.data *= np.take(idf, TF_IDFData.indices)
	# print "Done:\n",TF_IDFData
	# print "after transpose:\n", TF_IDFData.transpose()
	# saveSparse(array = TF_IDFData.transpose(), filename = filename) # saved is sparse of shape (N_samples, N_features)
	return TF_IDFData.transpose() # returned is a csc_matrix sparse of shape(N_samples, N_features)


def KNearest_Neighbour(X_train, y_train, X_test, y_test, num_neighbours):
	neigh = KNeighborsClassifier(n_neighbors=num_neighbours) # n neighbours = 1,3,5, default distance is standard Euclidean distance, weights in prediction is uniform
	neigh.fit(X=X_train, y=y_train)
	#print "K nearest neighbour prediction result:", neigh.predict(X_test)
	#print "True test label:", y_test
	return neigh.score(X= X_test, y=y_test)

def KNearest_Neighbour_Weighted_Similarity(X_train, y_train, X_test, y_test, num_neighbours):
	neigh = KNeighborsClassifier(n_neighbors=num_neighbours, weights = DistanceArrayToSimilarityArray)
	neigh.fit(X=X_train, y=y_train)
	#print "K nearest neighbour prediction result:", neigh.predict(X_test)
	#print "True test label:", y_test
	return neigh.score(X= X_test, y=y_test)

def DistanceArrayToSimilarityArray(arr):
	# print "Distance Array input is:\n", csc_matrix(arr) 
	global sigma
	# print "sigma in DistanceArrayToSimilarityArray function is:", sigma
	if(sigma == None):
		raise ValueError("Global sigma is not set!")
	K = np.exp(-1 * arr ** 2 / (2* float(sigma) ** 2)) # the larger the sigma, the closer K is to 1 
	# print K
	return K # k is of shape (N_sampels, N_samples)

#def DistanceValueToSimilarityValue(distance):
#	global sigma
#	if (sigma == None):
#		sigma = 500 # the best value for Raw data
#	return math.exp(-1*math.pow(distance,2)/(math.pow(sigma,2)*2))

def KNearestNeighbour_Analysis(data, label_column, num_folds, num_neighbours, weighted): # data is of shape ( N_samples, N_features), label_column is(N_samples, )
	# 5 fold or 10 fold
	accuracies = []
	kf = StratifiedKFold(label_column, n_folds=num_folds) # need stratified KF, otherwise, the normal KF just picks the first 80 as a subsample, 80-160 as another subsample -> in this case, the test data's label info won't be even in the training data set
	for train, test in kf:
		#print "train index:" , train
		#print "test index:", test
		X_train, X_test, y_train, y_test = data[train], data[test], label_column[train], label_column[test]
		#print "X_train:", X_train.shape
		#print "y_train:", y_train.shape
		#print "X_test:",  X_test.shape
		#print "y_test:", y_test.shape,"\n"
		if(weighted):
			accuracy  = KNearest_Neighbour_Weighted_Similarity(X_train, y_train, X_test, y_test, num_neighbours = num_neighbours)
		else:
			accuracy  = KNearest_Neighbour(X_train, y_train, X_test, y_test, num_neighbours = num_neighbours)
		# print "The accuracy:", accuracy
		accuracies.append(accuracy)
	global sigma
	print "The mean of the accuracies:", np.mean(accuracies), "; The standard Deviation of the accuracies:", np.std(accuracies, dtype=np.float64), "" if (sigma == None) else "with sigma tuned as {sigma}".format(sigma= sigma)
	return np.mean(accuracies)


def SVM_Linear (data_train, label_train, data_test, label_test, C):
	lin_svm = svm.LinearSVC(dual = False, C = C, penalty = 'l1')
	lin_svm.fit(data_train, label_train)
	return lin_svm.score(data_test, label_test)

def SVM_SVC(data_train, label_train, data_test, label_test, l1_penalty): # l1_penalty not used here
	svc_svm = svm.SVC(kernel='linear')
	#svc_svm = svm.SVC()
	svc_svm.fit(data_train, label_train)
	return svc_svm.score(data_test, label_test)

def SVM_Analysis (data, label_column, num_folds, SVMToUse, l1_penalty = 1.0):
	accuracies = []
	kf = StratifiedKFold(label_column, n_folds=num_folds) # need stratified KF, otherwise, the normal KF just picks the first 80 as a subsample, 80-160 as another subsample -> in this case, the test data's label info won't be even in the training data set
	for train, test in kf:
		#print "train index:" , train
		#print "test index:", test
		X_train, X_test, y_train, y_test = data[train], data[test], label_column[train], label_column[test]
		#print "X_train:", X_train.shape
		#print "y_train:", y_train.shape
		#print "X_test:",  X_test.shape
		#print "y_test:", y_test.shape,"\n"
		accuracy  = SVMToUse(X_train, y_train, X_test, y_test, l1_penalty)
		# print "The accuracy for one fold run:", accuracy
		accuracies.append(accuracy)
	global sigma
	print "The mean of the accuracies:", np.mean(accuracies), "; The standard Deviation of the accuracies:", np.std(accuracies, dtype=np.float64), "" if (sigma == None) else "with sigma tuned as {sigma}".format(sigma= sigma)
	return np.mean(accuracies)

def featureSelection_Linear_SVM (data, label, penalty): # data in shape(N_samples, N_feature), label is a 1D vector (N_samples,)
	svc_svm = svm.LinearSVC(penalty = 'l1', dual= False, C = penalty) # Tune the C for sparsity 
	svc_svm.fit(data, label)
	selected_data = svc_svm.transform(data)
	# print "The selected shape is:", selected_data.shape
	return svc_svm.coef_

def featureSelection_Correlation(data, label): # data is in (N_samples, N_features)
	label_features_r = np.zeros(data.shape[1])
	for i in range (0, data.shape[1]):
		r_matrix = np.corrcoef(label, data[:,i])
		#print "r_matrix:\n", r_matrix
		label_features_r[i] = r_matrix[0][1] # or [1][0] since symmetric
	print label_features_r**2
	print "Correlation square in decreasing sequence:\n", np.sort(label_features_r**2)[::-1]
	return np.argsort(label_features_r**2)[::-1]
	
def featureSelection_SingleVariablePerformance(data, label, analysis): # analysis could be string of 'KNN', 'SVM_SVC' or 'SVM_Linear', 5 fold default, 7 neighbors default for Knn
	rank = np.zeros(data.shape[1])
	for i in range (0, data.shape[1]):
		if(analysis == "KNN"):
			single_accuracy = KNearestNeighbour_Analysis(data = np.reshape(data[:,i], (data.shape[0], 1)), label_column = label, num_folds = 5, num_neighbours = 7, weighted = False)
		elif(analysis == "SVM_SVC"):
			single_accuracy = SVM_Analysis(data = np.reshape(data[:,i], (data.shape[0], 1)), label_column = label,num_folds = 5, SVMToUse = SVM_SVC)
		elif(analysis == "SVM_Linear"):
			single_accuracy = SVM_Analysis(data = np.reshape(data[:,i], (data.shape[0], 1)), label_column = label,num_folds = 5, SVMToUse = SVM_Linear)
		else:
			raise ValueError("unrecoginzed analysis methodolgy")
		rank[i] = single_accuracy
	# print "performance array:", rank
	unique_performance = np.unique(rank)[::-1]
	# print "different value in performance array", unique_performance
	performance_dict = {}
	for i in range (0, len(rank)):
		if(not(rank[i] in performance_dict)):
			performance_dict[rank[i]] = []
		performance_dict[rank[i]].append(i)
	return collections.OrderedDict(sorted(performance_dict.items())[::-1])

def ensembleClassificationWeightedVotes(data, label_column, num_folds, num_neighbours, C, data_categoricalAdded = None):
	accuracies = []
	kf = StratifiedKFold(label_column, n_folds=num_folds) # need stratified KF, otherwise, the normal KF just picks the first 80 as a subsample, 80-160 as another subsample -> in this case, the test data's label info won't be even in the training data set
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train], data[test], label_column[train], label_column[test]
		
		knn = KNeighborsClassifier(n_neighbors=num_neighbours) # n neighbours = 1,3,5, default distance is standard Euclidean distance, weights in prediction is uniform
		knn.fit(X=X_train, y=y_train)
		knn_label = knn.predict(X_test)
		print 'knn_label', knn_label

		lin_svm = svm.LinearSVC(dual = False, C = C, penalty = 'l1')
		lin_svm.fit(X_train, y_train)
		lin_svm_label = lin_svm.predict(X_test)
		print 'lin_svm_label', lin_svm_label

		svc_svm = svm.SVC(kernel='linear')
		svc_svm.fit(X_train, y_train)
		svc_svm_label = svc_svm.predict(X_test)
		print 'svc_svm_label', svc_svm_label

		flag = False
		dtc = DecisionTreeClassifier()
		if(data_categoricalAdded != None):
			dtc.fit(data_categoricalAdded[train], y_train)
			dtc_label = dtc.predict(data_categoricalAdded[test])
			flag = False
		else:
			dtc.fit(X_train, y_train)
			dtc_label = dtc.predict(X_test)
			flag = True
		print 'dtc_label', dtc_label


		accuracy = accuracy_score(y_test, weightedVotesLabel(flag, knn_label, lin_svm_label, svc_svm_label, dtc_label))
		print "Ensemble accuracy for one run in 5-fold:",accuracy 
		accuracies.append(accuracy)
	global sigma
	print "The mean of the accuracies:", np.mean(accuracies), "; The standard Deviation of the accuracies:", np.std(accuracies, dtype=np.float64), "" if (sigma == None) else "with sigma tuned as {sigma}".format(sigma= sigma)
	return np.mean(accuracies)

def weightedVotesLabel(isForBagOfDrug, knn_label, lin_svm_label, svc_svm_label,dtc_label): # when the weights do not differ much, it is just find the mode
	# for normalized bag of drugs:
	if(isForBagOfDrug):
		KNN_WEIGHT = 0.5
		LIN_SVM_WEIGHT = 0.539
		SVC_SVM_WEIGHT = 0.538
		DECISION_WEIGHT = 0.537
	else: # for normalized data numerical + the categorical features
		KNN_WEIGHT = 0.576
		LIN_SVM_WEIGHT = 0.611
		SVC_SVM_WEIGHT = 0.542
		DECISION_WEIGHT = 0.556

	WEIGHT_SUM = KNN_WEIGHT + LIN_SVM_WEIGHT + SVC_SVM_WEIGHT+ DECISION_WEIGHT
	
	weighted_probability = {
	0:KNN_WEIGHT/WEIGHT_SUM,
	1:LIN_SVM_WEIGHT/WEIGHT_SUM,
	2:SVC_SVM_WEIGHT/ WEIGHT_SUM,
	3:1- KNN_WEIGHT/WEIGHT_SUM - LIN_SVM_WEIGHT/WEIGHT_SUM - SVC_SVM_WEIGHT/ WEIGHT_SUM
	}

	print np.argsort([weighted_probability[0], weighted_probability[1], weighted_probability[2], weighted_probability[3]])[::-1]
	
	final_label = np.zeros(len(knn_label), dtype = int)
	label_pool = np.vstack((knn_label,lin_svm_label,svc_svm_label,dtc_label))
	for j in range(0, label_pool.shape[1]):
		label_candidates = np.unique(label_pool[:,j])

		#initialize the dict for label votes
		candidates_dict = {}
		for i in range(0, len(label_candidates)):
			candidates_dict[label_candidates[i]] = 0
		for i in range(0, label_pool.shape[0]):
			candidates_dict[label_pool[i][j]] += weighted_probability[i]
		final_label[j] = sorted(candidates_dict.items(), key=operator.itemgetter(1))[::-1][0][0]
	return final_label


def decision_tree_analysis(data, label_column, num_folds ):
	accuracies = []
	kf = StratifiedKFold(label_column, n_folds=num_folds)
	dtc = DecisionTreeClassifier()
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train], data[test], label_column[train], label_column[test]
		dtc.fit(X_train, y_train)
		accuracy  =  dtc.score(X_test, y_test)
		# print "The accuracy for a single fold run in DecisionTreeClassifier:", accuracy
		# print "\nrank of feature importances:", np.argsort(dtc.feature_importances_)[::-1]
		# print "importances in decreasing order:", dtc.feature_importances_
		accuracies.append(accuracy)
	global sigma
	print "The mean of the accuracies:", np.mean(accuracies), "; The standard Deviation of the accuracies:", np.std(accuracies, dtype=np.float64), "" if (sigma == None) else "with sigma tuned as {sigma}".format(sigma= sigma)
	return np.mean(accuracies)	

def main():
	# data_numerical = loadArray("data_numerical.npz")
	data_numerical = loadArray("data_numerical_age_quantized.npz")
	#label_readmission = loadArray("label_readmission.npz")
	label_readmission = loadArray("label_readmission_two_class.npz")

	print "percentage Number of readmissions:", len(label_readmission[label_readmission == 1])/ float(len(label_readmission))
	print "percentage Number of patients who did not readmit:", len(label_readmission[label_readmission == 0])/float(len(label_readmission))

	#Normalize
	normalized_data_numerical = preprocessing.normalize(X = data_numerical, axis = 0, norm ='l2')
	#Standardize
	standardizer = preprocessing.StandardScaler().fit(data_numerical)
	standardized_data_numerical = standardizer.transform(data_numerical)

	# for i in range (0, standardized_data_numerical.shape[1]):
	# 	print np.mean(standardized_data_numerical[:,i])
	# 	print np.std(standardized_data_numerical[:,i])

	#for i in range (0, normalized_data_numerical.shape[1]):
	#	print np.linalg.norm(normalized_data_numerical[:,i],2)

	#------------------K nearest Neighbour------------------
	print "--------------K nearest neighbours analysis on Numerical Data----------------"
	folds = [5]
	neighbours = [7,15,30]
	for i in range (0, len(folds)):
		for j in range(0, len(neighbours)):
			print "Normalized, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
			KNearestNeighbour_Analysis(data = normalized_data_numerical, label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j], weighted = False)
			print "Standardized, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
			KNearestNeighbour_Analysis(data = standardized_data_numerical, label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j], weighted = False)

	# print "----------Tuning sigma for weighted KNN on Normalized Numerical Data-------------" # Tuning for Knn on Numerical Data does not help much
	# sigma_accuracies = []
	# global sigma
	# for i in np.arange(0.001,0.01,0.002):
	# 	sigma = i
	# 	print "Normalized, num_folds = ", 5, ", num_neighbours = ", 7
	# 	sigma_accuracies.append(KNearestNeighbour_Analysis(data = normalized_data_numerical, label_column = label_readmission, num_folds = 5, num_neighbours = 7, weighted = True))
	# plt.plot(np.arange(0.001,0.01,0.002), sigma_accuracies)
	# plt.savefig("Tuning_sigma_normalized_numerical_data_sigma_0.001_to_0.01.png")
	# plt.show()

	#------------------- SVM ------------------------------
	print "------------------- SVM on Numerical Data------------------------------"
	print "SVM_Linear normalized_data_numerical"
	SVM_Analysis(data = normalized_data_numerical, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear)
	print "SVM_Linear standardized_data_numerical"
	SVM_Analysis(data = standardized_data_numerical, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear)
	print "SVM_SVC normalized data numerical"
	SVM_Analysis(data = normalized_data_numerical, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_SVC)
	print "SVM_SVC standardized_data_numerical"
	SVM_Analysis(data = standardized_data_numerical, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_SVC)



	data_bag_drugs = loadSparse("data_bagOfDrugs_sparse.npz")
	selected_column = []
	for i in range(0, data_bag_drugs.toarray().shape[1]):
		# print "if", i+1, "th column is all zeros:", (data_bag_drugs.toarray()[:,i] == 0).all() # 16th and 17th Drug are never taken, eliminate them from bag of drugs (examide & citoglipton)
		if(not (data_bag_drugs.toarray()[:,i] == 0).all()):
			selected_column.append(True)
		else:
			selected_column.append(False)
		#print "column sum is:", np.sum(data_bag_drugs.toarray()[:,i])

	print "after truncate the zero column features for bag of drugs:", data_bag_drugs[:,np.array(selected_column) == True].shape
	data_bag_drugs = data_bag_drugs[:,np.array(selected_column) == True]
	#Normalize
	normalized_bag_drugs = preprocessing.normalize(X = data_bag_drugs.toarray().astype(float), axis = 0, norm = 'l2')
	#standardize
	standardized_bag_drugs = preprocessing.StandardScaler().fit(data_bag_drugs.toarray().astype(float)).transform(data_bag_drugs.toarray().astype(float))
	#TFIDF
	tfidf_bag_drugs = TF_IDF(documentData = data_bag_drugs.transpose(), filename = "data_bagOfDrugs_TFIDF").toarray()
	# print (tfidf_bag_drugs.transpose().toarray() == tfidf_bag_drugs.toarray().T).all()
	# print tfidf_bag_drugs.shape


	#for i in range(0, normalized_bag_drugs.shape[1]):
	#	print i,"th column mean of standardized:", np.mean(standardized_bag_drugs[:,i])
	#	print i, "th column sd of standardized:", np.std(standardized_bag_drugs[:,i])
	#	print i, "th column norm:", np.linalg.norm(normalized_bag_drugs[:,i], 2)


	print "--------------K nearest neighbours analysis on Bag of Drugs----------------"
	folds = [5]
	neighbours = [7,15,30]
	for i in range (0, len(folds)):
		for j in range(0, len(neighbours)):
			print "Raw, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
			KNearestNeighbour_Analysis(data = data_bag_drugs.toarray().astype(float), label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j], weighted = False)
			print "Normalized, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
			KNearestNeighbour_Analysis(data = normalized_bag_drugs, label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j], weighted = False)
			print "Standardized, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
			KNearestNeighbour_Analysis(data = standardized_bag_drugs, label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j], weighted = False)
			print "TFIDF, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
			KNearestNeighbour_Analysis(data = tfidf_bag_drugs, label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j], weighted = False)


	# print "----------Tuning sigma for weighted KNN on Bag of Drugs-------------" # Tuning the weghts in Knn for Bag of Drugs improve a little accuracy : sigma = 0.005 - 0.494349798263 for 5 fold 7 neighbours
	# sigma_accuracies = []
	# global sigma
	# for i in np.arange(0.001, 0.01, 0.001):
	# 	sigma = i
	# 	print "Normalized, num_folds = ", 5, ", num_neighbours = ", 7
	# 	sigma_accuracies.append(KNearestNeighbour_Analysis(data = normalized_bag_drugs, label_column = label_readmission, num_folds = 5, num_neighbours = 7, weighted = True))
	# plt.plot(np.arange(0.001, 0.01, 0.001), sigma_accuracies)
	# plt.savefig("Tuning_sigma_normalized_bag_of_drugs_data_np.arange(0.001, 0.01, 0.001).png")
	# plt.show()

	print "------------------- SVM on Bag of Drugs Normalized------------------------------"
	print "SVM_Linear normalized bag drugs"
	SVM_Analysis(data = normalized_bag_drugs, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear) # for normalized bag of drugs, l2 is better than l1
	print "SVM_Linear raw bag drugs"
	SVM_Analysis(data = data_bag_drugs.toarray().astype(float), label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear) # raw bag of drugs, use l1
	print "SVM_Linear TFIDF bag drugs"
	SVM_Analysis(data = tfidf_bag_drugs, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear) # raw bag of drugs, use l1

	print "SVM_SVC normalized bag drugs"
	SVM_Analysis(data = normalized_bag_drugs, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_SVC)
	print "SVM_SVC raw bag drugs"
	SVM_Analysis(data = data_bag_drugs.toarray().astype(float), label_column = label_readmission,num_folds = 5, SVMToUse = SVM_SVC)
	print "SVM_SVC TFIDF bag drugs"
	SVM_Analysis(data = tfidf_bag_drugs, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_SVC)




	#-----------------------------Feature selection---------------------------------

	#---------------l1 SVM non zero coefficients on Numerical--------------
	# numerical_penalties = [0.05,0.07,0.08,0.15,0.16,1.0]
	numerical_penalties = [0.05,0.07,0.08,0.11,0.15, 1.0] # for age quantized
	optimal_scores = []
	feature_sets = []
	for i in range(0, len(numerical_penalties)):
		numerical_data_penalty = numerical_penalties[i] # {0.05: 3 features set, 0.07: 4 features set, 0.08: 5 features set, 0.15: 6 features set, 0.16: 7 features set, 1.0: 8 features set}
		print "\nnumerical_data_penalty = ", numerical_data_penalty
		feature_weights_numerical = featureSelection_Linear_SVM(data = normalized_data_numerical, label = label_readmission, penalty = numerical_data_penalty)
		# print "rank of numerical features:", feature_weights_numerical
		# print "Number of features in the set:", len(feature_weights_numerical[feature_weights_numerical!=0])
		print "feature set is:", np.array(range(0,8))[feature_weights_numerical[0]!=0]
		feature_sets.append(len(feature_weights_numerical[feature_weights_numerical!=0]))
	

		#RFE is the same as ranking the features based on the absolute value of coef_ given, might not be as good as tuning C in the LinearSVC model
		#estimator = svm.LinearSVC(penalty = 'l1', dual = False, C = numerical_data_penalty)
		#rfe = fs.RFE(estimator = estimator, n_features_to_select = 3)
		#rfe.fit(normalized_data_numerical, label_readmission)
		#print "rfe on numerical support_:",rfe.support_
		#print "rfe on numerical ranking_:", rfe.ranking_

		#RFECV tuned the number of features to include by cross validation, however, feature importance are still based on the absolute value of coef_; Could Tune the C to indicate how many features wanna include in the tuning
		lin_svm = svm.LinearSVC(penalty = 'l1', dual= False, C = numerical_data_penalty)
		rfecv = fs.RFECV(estimator=lin_svm, step=1, cv=StratifiedKFold(label_readmission, 5),
		              scoring='accuracy')
		rfecv.fit(normalized_data_numerical, label_readmission)
		print("Optimal number of features by RFECV : %d" % rfecv.n_features_)
		print "Optimal Set is:", np.array(range(0,8))[np.array(rfecv.support_) == True]
		# print "rankings:", rfecv.ranking_
		# print "supports:", rfecv.support_
		print "Optimal Score:",rfecv.grid_scores_[rfecv.n_features_-1]
		optimal_scores.append(rfecv.grid_scores_[rfecv.n_features_-1])
		# Plot number of features VS. cross-validation scores
		# plt.figure()
		# plt.xlabel("Number of features selected")
		# plt.ylabel("{n} Fold Cross validation score (nb of correct classifications)".format(n = 5))
		# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
		# plt.savefig("feature_selection/RFECV_numerical_{n}_featuresSet_C_{c}.png".format(n= len(feature_weights_numerical[feature_weights_numerical!=0]), c = numerical_data_penalty))
		# plt.show()
	
	print "last entry in optimal scores",optimal_scores[len(optimal_scores)-1]
	plt.plot(feature_sets, optimal_scores)
	# plt.plot(feature_sets, average_scores)
	plt.xlabel("number features in the set_age_quantized")
	plt.ylabel("optimal average accuracy")
	plt.savefig("feature_selection/numerical_age_quantized_optimal_accuracy_VS_num_featureSet.png")
	plt.clf()
	# plt.show()
	


    #---------------l1 SVM non zero coefficients on Bag of Drugs--------------
	bag_drugs_penalties = [0.001, 0.003, 0.004,0.006,0.01,0.03,0.04, 0.05,0.1,0.2,0.3,0.5,1.0] # raw
	# bag_drugs_penalties = [1.0]
	# #[0.01: 3, 0.003: 6, 0.004:7, 0.006:8, 0.01:9, 0.03:10, 0.04:11, 0.05:12, 0.1:13,0.2:15,0.3:17,0.5:20,1.0:21] penalty C VS number of drugs included in the set
	data_for_select = data_bag_drugs.toarray().astype(float)
	flag = 'raw'

	# bag_drugs_penalties = [0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,3.0] # normalized
	# data_for_select = normalized_bag_drugs
	# flag = 'normalized'

	optimal_scores = []
	optimal_features = []
	feature_sets = []
	average_scores = []
	for i in range(0, len(bag_drugs_penalties)):
		bag_drugs_penalty = bag_drugs_penalties[i]
		print "bag_drugs_penalty = ", bag_drugs_penalty
		feature_weights_bag_of_drugs = featureSelection_Linear_SVM(data = data_for_select, label = label_readmission, penalty = bag_drugs_penalty)
		# print "rank of drugs ({flag}):".format(flag = flag), feature_weights_bag_of_drugs
		# print "Number of features in the set:", len(feature_weights_bag_of_drugs[feature_weights_bag_of_drugs!=0])
		print "Selected Set:", np.array(range(0,21))[feature_weights_bag_of_drugs[0]!=0]
		feature_sets.append(len(feature_weights_bag_of_drugs[feature_weights_bag_of_drugs!=0]))

		# alternatively, just feed the SVM model with this selected set and compare the results, no need to use scikit learn model RFECV
		# average_accuracy = SVM_Analysis(data = data_for_select[:,np.array(range(0,21))[feature_weights_bag_of_drugs[0]!=0]], label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear, l1_penalty = bag_drugs_penalty) # raw bag of drugs, use l1
		# print "average accuracy with this features set:", average_accuracy
		# average_scores.append(average_accuracy)

		#RFECV tuned the number of features to include by cross validation, however, feature importance are still based on the absolute value of coef_; 
		lin_svm = svm.LinearSVC(penalty = 'l1', dual= False, C = bag_drugs_penalty)
		rfecv = fs.RFECV(estimator=lin_svm, step=1, cv=StratifiedKFold(label_readmission, 5),
		              scoring='accuracy')
		rfecv.fit(data_for_select, label_readmission) # raw
		print "Optimal Selected Index of the Drugs:", np.array(range(0,21))[np.array(rfecv.support_) == True]
		print "Optimal Grid Score:", rfecv.grid_scores_[rfecv.n_features_-1]
		print "avergae grid score with this set of drugs selected", np.mean(rfecv.grid_scores_[0:len(feature_weights_bag_of_drugs[feature_weights_bag_of_drugs!=0])])
		average_scores.append(np.mean(rfecv.grid_scores_[0:len(feature_weights_bag_of_drugs[feature_weights_bag_of_drugs!=0])]))
		optimal_scores.append(rfecv.grid_scores_[rfecv.n_features_-1])
		print("Optimal number of features by RFECV : %d" % rfecv.n_features_)
		optimal_features.append(rfecv.n_features_)
		print "rankings:", rfecv.ranking_
		print "supports:", rfecv.support_

		#Plot number of features VS. cross-validation scores
		# plt.figure()
		# plt.xlabel("Number of Drugs selected_{flag}".format(flag = flag))
		# plt.ylabel("{n} Fold Cross validation score (nb of correct classifications)".format(n = 5))
		# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
		# plt.ylim([0.52,0.56])
		# plt.savefig("feature_selection/RFECV_bag_of_drugs_{flag}_{n}_featuresSet_C_{c}_optimal_{k}.png".format(flag = flag, n= len(feature_weights_bag_of_drugs[feature_weights_bag_of_drugs!=0]), c = bag_drugs_penalty, k = rfecv.n_features_))
		# plt.show()
		# plt.clf()
	

	# Compare the optimal accuracies; Notice that For Raw bag of drugs, the 2nd last and 3rd last run gives the best accuracy of 0.541271161176 by keeping the [ 1  8 11 12 14 17] or [0 1 8 11 12 14 15 17 ] drugs
	# For normlized bag of drugs: keep 8 drugs in the feature set gives the best accuracy of 0.540101797498 by keeping  [ 0  1  6  9 10 11 15] in the set
	# For numerical data, best accuracy achieved by keeping all 8 numerical data
	# plt.plot(range(1, len(optimal_scores)+1), optimal_scores)
	plt.plot(feature_sets, optimal_scores)
	# plt.plot(feature_sets, average_scores)
	plt.xlabel("number features in the set")
	plt.ylabel("average accuracy")
	plt.savefig("feature_selection/bag_of_drugs_optimal_accuracy_VS_num_featureSet_{flag}.png".format(flag = flag))
	plt.clf()
	# plt.savefig("feature_selection/numerical_optimal_accuracy_VS_num_featureSet.png")
	# plt.show()


	#---------------variable ranking: Correlation--------------------
	# Numerical Data, decreasing order: [7 6 5 1 4 3 2 0], same for normalize and standardized
	print featureSelection_Correlation(data = normalized_data_numerical, label = label_readmission)
	# print featureSelection_Correlation(data = standardized_data_numerical, label = label_readmission)

	# Bag of Drugs, decreasing order: [15  0  1  6 11  9 10 12 14  2 19  4  7 17  8 18  5 20 16 13  3], same for raw , normalized and standardized
	print featureSelection_Correlation(data = normalized_bag_drugs, label = label_readmission)
	# print featureSelection_Correlation(data = data_bag_drugs.toarray().astype(float), label = label_readmission)
	# print featureSelection_Correlation(data = standardized_bag_drugs, label = label_readmission)


	#---------------variable ranking: Chi square--------------------
	# Numerical Data ranking: 5,6,7 > 1,3 > 2,4 > 0 same ranking as l1 based SVM method
	print "chi_sq_selector on data numerical"
	chi_sq_selector = fs.SelectKBest(fs.chi2, k=5).fit(normalized_data_numerical, label_readmission)
	# print chi_sq_selector.get_support(indices=True)
	print np.argsort(chi_sq_selector.scores_)[::-1]
	# print chi_sq_selector.scores_[np.argsort(chi_sq_selector.scores_)[::-1]]
	print np.sort(chi_sq_selector.scores_)[::-1]

	# Normalized Bag of Drug data the selection is slightly different from l1 based SVM method
	print "chi_sq_selector on normalized_bag_drugs" # ranking: [11  1 19 18  5 20 12  0 14 15 17  6 13  8 10  9  2 16  4  7  3]
	chi_sq_selector = fs.SelectKBest(fs.chi2, k=10).fit(normalized_bag_drugs, label_readmission) # X input must be non-negative, thus, standardized data can not be used here
	# print chi_sq_selector.get_support(indices=True)
	print np.argsort(chi_sq_selector.scores_)[::-1]
	print np.sort(chi_sq_selector.scores_)[::-1]
	
	# Raw bag of drug selection is similar to that of l1 SVM method
	print "chi_sq_selector on raw_bag_drugs" # ranking: [15  0  1  6 11  9 10 12 14  2 19  4  7 17  8  5 18 20 16 13  3]
	chi_sq_selector = fs.SelectKBest(fs.chi2, k=10).fit(data_bag_drugs.toarray().astype(float), label_readmission)
	# print chi_sq_selector.get_support(indices=True)
	print np.argsort(chi_sq_selector.scores_)[::-1]
	print np.sort(chi_sq_selector.scores_)[::-1]
	
	#---------------variable ranking: Single feature Knn performance------------
	# Numerical, OrderedDict([(0.55329884698368959, [5]), (0.52759201386961918, [7]), (0.51347227480024271, [4]), (0.51190984770272352, [1]), (0.50994460160897082, [2]), (0.50538470063241681, [0]), (0.49359315016821775, [3]), (0.47728056901021559, [6])])	
	# print np.unique(normalized_data_numerical[:,0])
	print featureSelection_SingleVariablePerformance(data = normalized_data_numerical, label = label_readmission, analysis = "KNN")
	
	# Normalized Bag of Drugs, [12 | 17 | 20 18  5 19 | 16 | 0 | 1 | 11 | 4 | 2 | 15 | 14 | 8 | 13 | 9 | 6 | 7 | 10 | 3]
	print featureSelection_SingleVariablePerformance(data = normalized_bag_drugs, label = label_readmission, analysis = "KNN")
	
	# Raw Bag of Drugs, same as normalized bag of drugs: OrderedDict([(0.53920759394477735, [12]), (0.5391486350604382, [17]), (0.53911915585958892, [5, 18, 19, 20]), (0.53865729637833237, [16]), (0.53381307351705176, [0]), (0.52489104452855595, [1]), (0.52381996866933833, [11]), (0.5232598546812286, [4]), (0.52277834159475201, [2]), (0.51957405803809131, [15]), (0.47659274593528222, [14]), (0.47655343824210672, [8]), (0.47652396097248345, [13]), (0.4637502429830197, [9]), (0.46333742167737402, [6]), (0.46262997203310513, [7]), (0.46257096245652402, [10]), (0.46086119198358694, [3])])
	# print "Raw Bag of Drugs Single feature performance KNN"
	# print featureSelection_SingleVariablePerformance(data = data_bag_drugs.toarray().astype(float), label = label_readmission, analysis = "KNN")

	#---------------variable ranking: Single feature Linear SVM performance------------
	# Numerical, [7 6 5  3/0 2 1 4], with [0,3] same performance ; OrderedDict([(0.59553309710983116, [7]), (0.56933579536811496, [6]), (0.55458617816804079, [5]), (0.53911915585958892, [0, 3]), (0.53793996755039686, [2]), (0.53692781533223133, [1]), (0.53172973101867327, [4])])
	print featureSelection_SingleVariablePerformance(data = normalized_data_numerical, label = label_readmission, analysis = "SVM_Linear")
	
	# Normalized Bag of drugs, OrderedDict([(0.54060294390951324, [1]), (0.53968910751365728, [11]), (0.53921742098880254, [12]), (0.53911915585958892, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20]), (0.53910932978122417, [17])])
	print featureSelection_SingleVariablePerformance(data = normalized_bag_drugs, label = label_readmission, analysis = "SVM_Linear")
	
	# Raw Bag of Durgs, OrderedDict([(0.54060294390951324, [1]), (0.53968910751365728, [11]), (0.53921742098880254, [12]), (0.5391486350604382, [17]), (0.53911915585958892, [0, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20]), (0.5390208950754678, [3]), (0.53860816308952764, [2])])
	# print "Raw bag of Drugs single feature performance SVM_Linear"
	# print featureSelection_SingleVariablePerformance(data = data_bag_drugs.toarray().astype(float), label = label_readmission, analysis = "SVM_Linear")


	#-----------------------------Run Analysis on Selected Features--------------------------------------
	
	# print "--------------Knn on the selected significant features-------------"
	# numerical_selected = [7,6,5,1]
	# KNearestNeighbour_Analysis(data = normalized_data_numerical[:,numerical_selected], label_column = label_readmission, num_folds = 5, num_neighbours = 7, weighted = False)
	# raw_bag_of_drugs_selected = [8,11,12,14,17]
	# KNearestNeighbour_Analysis(data = data_bag_drugs.toarray().astype(float)[:,raw_bag_of_drugs_selected], label_column = label_readmission, num_folds = 5, num_neighbours = 7, weighted = False)

	print "-------------Comparison of Analysis with and without Age factor-----------"
	print normalized_data_numerical[:,1:].shape
	
	print "Knn with age"
	KNearestNeighbour_Analysis(data = normalized_data_numerical, label_column = label_readmission, num_folds = 5, num_neighbours = 7, weighted = False)
	print "Knn without age"
	KNearestNeighbour_Analysis(data = normalized_data_numerical[:,1:], label_column = label_readmission, num_folds = 5, num_neighbours = 7, weighted = False)
	print "Linear SVM with age"
	SVM_Analysis(data = normalized_data_numerical, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear)
	print "Linear SVM without age"
	SVM_Analysis(data = normalized_data_numerical[:,1:], label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear)
	print  "SVC SVM with age"
	SVM_Analysis(data = normalized_data_numerical, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_SVC)
	print  "SVC SVM without age"
	SVM_Analysis(data = normalized_data_numerical[:,1:], label_column = label_readmission,num_folds = 5, SVMToUse = SVM_SVC)
	
	data_categorical = loadArray("data_categorical.npz")
	combined_data_numbers = np.concatenate((data_numerical , data_categorical), axis = 1)
	print  "Decision Tree with age"
	decision_tree_analysis(data = combined_data_numbers, label_column = label_readmission, num_folds = 5)
	print  "Decision Tree without age"
	decision_tree_analysis(data = combined_data_numbers[:,1:], label_column = label_readmission, num_folds = 5)


	# print weightedVotesLabel(False, np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]), np.array([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]), np.array([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]), np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]))

	#---------------------------------------Ensemble models-------------------------------------------------------
	print "--------------------------Ensemble models------------------------"
	ensembleClassificationWeightedVotes(data = normalized_data_numerical, label_column= label_readmission, num_folds = 5, num_neighbours = 7, C = 1.0, data_categoricalAdded = np.concatenate((data_numerical , loadArray("data_categorical.npz")), axis = 1))
	ensembleClassificationWeightedVotes(data = normalized_bag_drugs, label_column= label_readmission, num_folds = 5, num_neighbours = 7, C = 1.0)

	#-----------------------------------Dicision Tree----------------------------
	data_categorical = loadArray("data_categorical.npz")
	combined_data_numbers = np.concatenate((data_numerical , data_categorical), axis = 1)
	print decision_tree_analysis(data = data_numerical, label_column = label_readmission, num_folds = 5)
	print decision_tree_analysis(data = normalized_data_numerical, label_column = label_readmission, num_folds = 5)
	print decision_tree_analysis(data = standardized_data_numerical, label_column = label_readmission, num_folds = 5)
	print decision_tree_analysis(data = combined_data_numbers, label_column = label_readmission, num_folds = 5)

	print decision_tree_analysis(data = data_bag_drugs.toarray().astype(np.float32), label_column = label_readmission, num_folds = 5)
	# print decision_tree_analysis(data = normalized_bag_drugs, label_column = label_readmission, num_folds = 5) # exactly the same as the raw bag of drugs
	# print decision_tree_analysis(data = tfidf_bag_drugs, label_column = label_readmission, num_folds = 5) #excatly the same as the raw bag of drugs

	return


if __name__ == "__main__":
	main()
# 