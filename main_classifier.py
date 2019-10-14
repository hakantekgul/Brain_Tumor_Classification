import cv2
from pathlib import Path
import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from classifiers import svm_classifier, neural_net, knn_classifier

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

def show_images(X,title):
	fig, axes = plt.subplots(5,5,figsize=(9,9),
		subplot_kw={'xticks':[], 'yticks':[]},
		gridspec_kw=dict(hspace=0.01, wspace=0.01))
	for i, ax in enumerate(axes.flat):
		ax.imshow(X[i],cmap='gray')
	plt.title(title)
	plt.show()

print('######## Starting Experiments ###############')
print('Please enter the folder name to read the images from:')
folder_name = input()
print(folder_name)
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
#show_images(X,'Read Images')

# Show the dimensionality reduction #
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X_flat)
approximation = pca.inverse_transform(X_pca)
approximation = approximation.reshape((nsamples,nx,ny))
#show_images(approximation,'Reduced Images')

# Start applying supervised learning # 
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

neural_net(X_train,y_train,X_test,y_test,learning_rate=1e-04,plotting=False)
# Now we have normalized datasets. We apply PCA. Note tha PCA above was just for visualization.
pca = PCA(n_components = 0.9)

pca.fit(X_train)

X_train_reduced = pca.transform(X_train)
X_test_reduced = pca.transform(X_test)

svm_classifier(X_train_reduced,y_train,X_test_reduced,y_test,'rbf',gamma='auto',plotting=False)
neural_net(X_train_reduced,y_train,X_test_reduced,y_test,learning_rate=1e-04,plotting=False)
knn_classifier(X_train,y_train,X_test,y_test,neighbors=10,plotting=False)
