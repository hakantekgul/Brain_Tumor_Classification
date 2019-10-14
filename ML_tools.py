import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
import itertools

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.2,1.0,10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

def plot_validation_curve(X,y,estimator,name): 
	if name == 'svm':
		param_range = np.logspace(-2, 1, 10)
		param_name = 'C'
	if name == 'dtree':
		param_range = np.arange(1,30)
		param_name = 'max_depth'
	if name == 'knn':
		param_range = np.arange(1,8)
		param_name = 'n_neighbors'
	if name == 'neural_net':
		param_range = [1e-05,1e-04,1e-03,1e-02,1e-01,1,2,3]
		param_name = 'alpha'
	if name == 'boosting':
		param_range = np.arange(50,200)
		param_name = 'n_estimators'

	train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=5, scoring="accuracy", n_jobs=1)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.title("Validation Curve for " + name)
	plt.xlabel(param_name)
	plt.ylabel("Score")
	plt.ylim(0.0, 1.1)
	plt.grid()
	lw = 2
	plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
	plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
	plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
	plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
	plt.legend(loc="best")
	plt.show()

def confusion_matrix_2(cm, classes,title='Confusion matrix', cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def confusion(Y_test,pred,title):
	cnf_matrix = confusion_matrix(Y_test,pred)
	plt.figure()
	a = confusion_matrix_2(cnf_matrix,classes=['0','1'],title=title)