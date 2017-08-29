# CSCI-599 Take-home exam
import sys
import numpy as np
from sklearn.decomposition import PCA

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":
	# Read in the arguments
	K = int(sys.argv[1])
	D = int(sys.argv[2])
	N = int(sys.argv[3])
	PATH_TO_DATA = sys.argv[4]
	
	# Import dataset
	dataDict = unpickle(PATH_TO_DATA)
	print(dataDict.keys())
	data = dataDict[b'data']
	labels = dataDict[b'labels']
	data = data[0:1000, :]
	labels = labels[0:1000]
	# Convert the RGB images to grayscale
	dataL = 0.299 * data[:, 0:1024] + 0.587 * data[:, 1024:2048] + 0.114 * data[:, 2048:3072]
	testSet = dataL[0:N, :]
	trainSet = dataL[N:, :]
	testLabel = labels[0:N]
	trainLabel = labels[N:]
#	print(testLabel)
#	print(len(testLabel))
#	print(len(trainLabel))
	testNum = N
	trainNum = 1000 - N
	
	# Apply PCA 
	pca = PCA(n_components=D, svd_solver='full')
	pca.fit(trainSet)
	train_pca = pca.transform(trainSet)
	test_pca = pca.transform(testSet)
	print(train_pca.shape, test_pca.shape)
	
	# Implement K-Nearest Neighbors classifier
	
	

