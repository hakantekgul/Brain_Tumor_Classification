import pandas as pd
import numpy as np
from sklearn import svm, datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt
from ML_tools import plot_learning_curve,plot_validation_curve,confusion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import time

def svm_classifier(X_train,y_train,X_test,y_test,kernel,gamma,plotting):
	# Apply SVM
	clf = svm.SVC(gamma='auto',random_state=3169,kernel=kernel,C=2)
	st = time.time()
	clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with SVM'
	clf_name = 'svm'
	print(str(clf.get_params))
	if plotting == True:
		plot_learning_curve(clf,'Learning Curve for SVM', X_train, y_train, (0.7, 1.01), n_jobs=5)
		plot_validation_curve(X_train,y_train,clf,clf_name)
		confusion(y_test,y_pred,title)

	print('Accuracy for SVM classifier is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for SVM classifier: ' + str(train_time) + ' seconds')
	print('Testing time for SVM classifier: ' + str(endt-stt) + ' seconds')


def tuned_svm_classifier(X_train,y_train,X_test,y_test,kernel,gamma,plotting):
	# Apply SVM
	clf = svm.SVC(random_state=3169,kernel='linear',gamma='auto')
	param_range = [0.01,0.25,0.5,0.75,1,1.25,1.5,2.75,3]
	clf = GridSearchCV(clf, param_grid={'C' : param_range}, cv = 5)
	st = time.time()
	clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with SVM'
	clf_name = 'svm'
	if plotting == True:
		plot_learning_curve(clf,'Learning Curve for SVM', X_train, y_train, (0.7, 1.01), n_jobs=4)
		confusion(y_test,y_pred,title)

	print('Accuracy for Tuned SVM classifier is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for Tuned SVM classifier: ' + str(train_time) + ' seconds')
	print('Testing time for Tuned SVM classifier: ' + str(endt-stt) + ' seconds')
	print('Best parameters: ' + str(clf.best_params_))
	print()
	return clf.best_params_


def dtree_classifier(X_train,y_train,X_test,y_test,plotting):
	
	clf = DecisionTreeClassifier(random_state = 3169, max_depth=3)
	st = time.time()
	clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with Decision Tree'
	#plot_tree(clf.fit(X_train, y_train),filled=True)
	plt.show()
	clf_name = 'dtree'
	if plotting == True:
		plot_learning_curve(clf,'Learning Curve for Decision Tree', X_train, y_train, (0.7, 1.01), n_jobs=4)
		plot_validation_curve(X_train,y_train,clf,clf_name)
		confusion(y_test,y_pred,title)

	print('Accuracy for Decision Tree classifier is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for Decision Tree classifier: ' + str(train_time) + ' seconds')
	print('Testing time for Decision Tree classifier: ' + str(endt-stt) + ' seconds')
	print()


def tuned_dtree_classifier(X_train,y_train,X_test,y_test,plotting):
	clf = DecisionTreeClassifier(random_state = 3169)
	param_range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	clf = GridSearchCV(clf, param_grid={'max_depth' : param_range}, cv = 5)
	st = time.time()
	clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with Tuned Decision Tree'
	#tree.plot_tree(clf.fit(X_train, Y_train)) 
	clf_name = 'dtree'
	if plotting == True:
		plot_learning_curve(clf,'Learning Curve for Tuned Decision Tree', X_train, y_train, (0.7, 1.01), n_jobs=4)
		confusion(y_test,y_pred,title)

	print('Accuracy for Tuned Decision Tree classifier is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for Tuned Decision Tree classifier: ' + str(train_time) + ' seconds')
	print('Testing time for Tuned Decision Tree classifier: ' + str(endt-stt) + ' seconds')
	print('Best parameters: ' + str(clf.best_params_))
	print()
	return clf.best_params_


def knn_classifier(X_train,y_train,X_test,y_test,neighbors,plotting):
	clf = KNeighborsClassifier(n_neighbors=neighbors,weights='distance')
	st = time.time()
	clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with KNN'
	clf_name = 'knn'
	if plotting == True:
		plot_learning_curve(clf,'Learning Curve for KNN', X_train, y_train, (0.7, 1.01), n_jobs=4)
		plot_validation_curve(X_train,y_train,clf,clf_name)
		confusion(y_test,y_pred,title)

	print('Accuracy for KNN classifier is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for KNN classifier: ' + str(train_time) + ' seconds')
	print('Testing time for KNN SVM classifier: ' + str(endt-stt) + ' seconds')
	print()

def tuned_knn_classifier(X_train,y_train,X_test,y_test,neighbors,plotting):
	clf = KNeighborsClassifier()
	param_range = np.arange(1,8)
	clf = GridSearchCV(clf, param_grid={'n_neighbors' : param_range}, cv=5)
	st = time.time()
	clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with Tuned KNN'
	clf_name = 'knn'
	if plotting == True:
		plot_learning_curve(clf,'Learning Curve for Tuned KNN', X_train, y_train, (0.7, 1.01), n_jobs=4)
		confusion(y_test,y_pred,title)

	print('Accuracy for Tuned KNN classifier is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for Tuned KNN classifier: ' + str(train_time) + ' seconds')
	print('Testing time for Tuned KNN classifier: ' + str(endt-stt) + ' seconds')
	print('Best parameters: ' + str(clf.best_params_))
	print()
	return clf.best_params_

def tuned_weighted_knn_classifier(X_train,y_train,X_test,y_test,neighbors,plotting):
	clf = KNeighborsClassifier(weights='distance')
	param_range = np.arange(1,8)
	clf = GridSearchCV(clf, param_grid={'n_neighbors' : param_range}, cv=5)
	st = time.time()
	clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with Tuned Weighted KNN'
	clf_name = 'knn'
	if plotting == True:
		plot_learning_curve(clf,'Learning Curve for Tuned Weighted KNN', X_train, y_train, (0.7, 1.01), n_jobs=4)
		confusion(y_test,y_pred,title)

	print('Accuracy for Tuned Weighted KNN classifier is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for Tuned Weighed KNN classifier: ' + str(train_time) + ' seconds')
	print('Testing time for Tuned Weighed KNN classifier: ' + str(endt-stt) + ' seconds')
	print('Best parameters: ' + str(clf.best_params_))
	print()
	return clf.best_params_


def neural_net(X_train,y_train,X_test,y_test,learning_rate,plotting):
	clf = MLPClassifier(activation='relu',alpha=1e-04,batch_size='auto',learning_rate='adaptive',learning_rate_init=learning_rate,solver='adam',random_state=3169)
	st = time.time()
	clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with Neural Network'
	clf_name = 'neural_net'
	if plotting == True:
		plot_learning_curve(clf,'Learning Curve for Neural Network', X_train, y_train, (0.7, 1.01), n_jobs=4)
		plot_validation_curve(X_train,y_train,clf,clf_name)
		confusion(y_test,y_pred,title)

	print('Accuracy for Neural Network is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for Neural Network: ' + str(train_time) + ' seconds')
	print('Testing time for Neural Network: ' + str(endt-stt) + ' seconds')
	print()

def tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate,plotting):
	clf = MLPClassifier(activation='relu',max_iter=5000,alpha=1e-05,batch_size='auto',learning_rate='adaptive',learning_rate_init=learning_rate,solver='adam')
	alpha_range = [1e-05,1e-04,1e-03,1e-02,1e-01,1,2,3]
	lr_range = [0.00001,0.0001,0.001,0.01,0.1,0.5,0.8,1]
	params = {'alpha' : alpha_range, 'learning_rate_init' : lr_range, 'activation': ['relu','tanh']}
	clf = GridSearchCV(clf, param_grid=params, cv=5)
	st = time.time()
	clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with Tuned Neural Network'
	clf_name = 'neural_net'
	if plotting == True:
		plot_learning_curve(clf,'Learning Curve for Neural Network', X_train, y_train, (0.7, 1.01), n_jobs=4)
		confusion(y_test,y_pred,title)
		clf_best = MLPClassifier(hidden_layer_sizes=(5, 2), random_state = 3169, max_iter = 1)
		clf_best.set_params(alpha=clf.best_params_['alpha'], learning_rate_init=clf.best_params_['learning_rate_init'])
		num_epochs = 1000
		train_loss = np.zeros(num_epochs)
		train_scores = np.zeros(num_epochs)
		val_scores = np.zeros(num_epochs)
		# Split training set into training and validation
		X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=3169)
		for i in range(num_epochs):
			clf_best.fit(X_train_, y_train_)
			train_loss[i] = clf_best.loss_
			train_scores[i] = accuracy_score(y_train_, clf_best.predict(X_train_))
		range_loss = np.arange(num_epochs) + 1
		plt.figure()
		plt.plot(range_loss, train_loss)
		plt.title('Training loss curve for neural network')
		plt.xlabel('Epochs')
		plt.ylabel("Loss")
		plt.grid()
		plt.show()

	print('Accuracy for Tuned Neural Network is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for Tuned NN: ' + str(train_time) + ' seconds')
	print('Testing time for Tuned NN: ' + str(endt-stt) + ' seconds')
	print('Best parameters: ' + str(clf.best_params_))
	print()
	return clf.best_params_


def boost_classifier(X_train,y_train,X_test,y_test,plotting):
	clf = DecisionTreeClassifier(random_state = 3169,max_depth=2,min_samples_leaf=1)
	boost_clf = AdaBoostClassifier(base_estimator = clf, random_state=3169, n_estimators=100)
	st = time.time()
	boost_clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = boost_clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with Boosting'
	#plot_tree(clf.fit(X_train, y_train),filled=True)
	plt.show()
	clf_name = 'boosting'
	if plotting == True:
		plot_learning_curve(boost_clf,'Learning Curve for Boosting', X_train, y_train, (0.7, 1.01), n_jobs=4)
		plot_validation_curve(X_train,y_train,boost_clf,clf_name)
		confusion(y_test,y_pred,title)

	print('Accuracy for Boosting classifier is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for Boosting classifier: ' + str(train_time) + ' seconds')
	print('Testing time for Boosting classifier: ' + str(endt-stt) + ' seconds')	
	print()

def tuned_boosting_classifier(X_train,y_train,X_test,y_test,plotting):
	clf = DecisionTreeClassifier(random_state = 3169,max_depth=2,min_samples_leaf=1)
	boost_clf = AdaBoostClassifier(base_estimator = clf, random_state=3169)
	param_range = np.arange(10,100)
	boost_clf = GridSearchCV(boost_clf, param_grid={'n_estimators' : param_range}, cv = 5)
	st = time.time()
	boost_clf.fit(X_train,y_train)
	end = time.time()
	train_time = end-st
	stt = time.time()
	y_pred = boost_clf.predict(X_test)
	endt = time.time()
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	title = 'Confusion Matrix with Tuned Boosting'
	#tree.plot_tree(clf.fit(X_train, Y_train)) 
	clf_name = 'boosting'
	if plotting == True:
		plot_learning_curve(boost_clf,'Learning Curve for Tuned Boosting classifier', X_train, y_train, (0.7, 1.01), n_jobs=4)
		confusion(y_test,y_pred,title)

	print('Accuracy for Tuned Boosting classifier is ' + str(accuracy_score(y_test,y_pred)))
	print('Training time for Tuned Boosting classifier: ' + str(train_time) + ' seconds')
	print('Testing time for Tuned Boosting classifier: ' + str(endt-stt) + ' seconds')
	print('Best parameters: ' + str(boost_clf.best_params_))
	print()
	return boost_clf.best_params_



	