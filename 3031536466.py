# CSCI-599 Take-home exam
import sys
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance

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
	testNum = N
	trainNum = 1000 - N
	
	# Apply PCA 
	pca = PCA(n_components=D, svd_solver='full')
	pca.fit(trainSet)
	train_pca = pca.transform(trainSet)
	test_pca = pca.transform(testSet)
	
	# Implement K-Nearest Neighbors classifier
	distArr = distance.cdist(test_pca, train_pca, 'euclidean')
	neighbors = np.argsort(distArr, axis=1)[:, 0:K]
	predictLabel = []
	for n in range(testNum):
		voteCt = np.zeros((10))
		for k in range(K):
			voteCt[trainLabel[neighbors[n,k]]] += (1/distArr[n, neighbors[n,k]])
		predictLabel.append(np.argmax(voteCt))	
	
	# Write the output file	
	with open('3031536466.txt', 'w') as fd:
		for n in range(testNum):
			fd.write(str(predictLabel[n]) + ' ' + str(testLabel[n]) + '\n')
