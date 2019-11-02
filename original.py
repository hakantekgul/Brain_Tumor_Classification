import cv2
from pathlib import Path
import glob
import numpy as np
import scipy.ndimage as ndimage
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from classifiers import svm_classifier, neural_net, knn_classifier

def original(sigma,folder_name,show=False):
	# load the images 
	def load_positive(path):
		filenames = glob.glob(path)
		images = [cv2.imread(img) for img in filenames]
		positives = []
		for img in images:
			imge = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			positives.append(imge)
		return np.array(positives)

	def load_negative(path): 
		filenames = glob.glob(path)
		images = [cv2.imread(img) for img in filenames]
		negatives = []
		for img in images:
			imge = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			negatives.append(imge)
		return np.array(negatives)

	def show_images(X):
		fig, axes = plt.subplots(5,5,figsize=(9,9),
			subplot_kw={'xticks':[], 'yticks':[]},
			gridspec_kw=dict(hspace=0.01, wspace=0.01))
		for i, ax in enumerate(axes.flat):
			ax.imshow(X[i],cmap='gray')
		plt.show()

	yes = load_positive(folder_name+"/yes/*.png")
	no = load_negative(folder_name+"/no/*.png")

	print('Number of Positive Images: ' + str(yes.shape[0]))
	print('Number of Negative Images: ' + str(no.shape[0]))

	yes_labels = np.ones(yes.shape[0])
	no_labels = np.zeros(no.shape[0])

	X = np.vstack((yes,no))
	y = np.hstack((yes_labels,no_labels))


	nsamples, nx, ny = X.shape
	X_flat = X.reshape((nsamples,nx*ny))
	print(X_flat.shape)
	X_train, X_test, y_train, y_test = train_test_split(X_flat,y,test_size=0.3,random_state=0)
	if show == True:
		show_images(X,'Read Images')

	# Show the dimensionality reduction #
	pca = PCA(n_components=0.95)
	X_pca = pca.fit_transform(X_flat)
	approximation = pca.inverse_transform(X_pca)
	approximation = approximation.reshape((nsamples,nx,ny))
	if show == True:
		show_images(approximation,'Reduced Images')

	# Start applying supervised learning # 
	scaler = StandardScaler()

	# Fit on training set only.
	scaler.fit(X_train)

	# Apply transform to both the training set and the test set.
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	predictions_svm = svm_classifier(X_train,y_train,X_test,y_test,'rbf',gamma='auto',plotting=False)
	predictions_nn = neural_net(X_train,y_train,X_test,y_test,learning_rate=1e-04,plotting=False)
	pca = PCA(n_components = 0.95)

	pca.fit(X_train)

	X_train_reduced = pca.transform(X_train)
	X_test_reduced = pca.transform(X_test)

	predictions_svm2 = svm_classifier(X_train_reduced,y_train,X_test_reduced,y_test,'rbf',gamma='auto',plotting=False)
	predictions_nn2 = neural_net(X_train_reduced,y_train,X_test_reduced,y_test,learning_rate=1e-04,plotting=False)
	predictions_knn = knn_classifier(X_train,y_train,X_test,y_test,neighbors=10,plotting=False)

	return predictions_nn, predictions_nn2, predictions_svm, predictions_svm2, predictions_knn, y_test


