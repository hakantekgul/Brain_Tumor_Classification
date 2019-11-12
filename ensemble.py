import cv2
from pathlib import Path
import glob
import numpy as np
import scipy.ndimage as ndimage
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from classifiers import svm_classifier, neural_net, knn_classifier
from gaussian_hpf import gaussian_hpf
from gaussian_lpf_hpf import gaussian_lpf_hpf
from gaussian_lpf import gaussian_lpf
from original import original
from median_filter import median_filter
from collections import Counter
from sklearn.metrics import accuracy_score
import sys
from ML_tools import confusion


print('######## Starting Experiments ###############')
'''
print('Please enter the folder name to read the images from:')
folder_name = input()

print('ORIGINAL:')
predictions_nn, predictions_nn2, predictions_svm, predictions_svm2, predictions_knn, y_test = original(0,folder_name)
print('LPF:')
predictions_nn_lpf, predictions_nn2_lpf, predictions_svm_lpf, predictions_svm2_lpf, predictions_knn_lpf = gaussian_lpf(20,folder_name)
print('HPF:')
predictions_nn_hpf, predictions_nn2_hpf, predictions_svm_hpf, predictions_svm2_hpf, predictions_knn_hpf = gaussian_hpf(20,folder_name)
print('LPF+HPF:')
predictions_nn_lpf_hpf, predictions_nn2_lpf_hpf, predictions_svm_lpf_hpf, predictions_svm2_lpf_hpf, predictions_knn_lpf_hpf = gaussian_lpf_hpf(20,folder_name)
print('Median:')
predictions_nn_median, predictions_nn2_median, predictions_svm_median, predictions_svm2_median, predictions_knn_median = median_filter(20,folder_name)
'''

# FINAL ENSEMBLE LEARNING with good accuracy results 
print('EQUALIZED DATA: ')
y_test = original(0,'equalized_data')
print('LPF:')
predictions_svm_lpf = gaussian_lpf(2,'equalized_data')
print('HPF:')
predictions_svm_hpf = gaussian_hpf(15,'equalized_data')
print('LPF+HPF:')
predictions_svm_lpf_hpf = gaussian_lpf_hpf(15,'equalized_data')
print('MEDIAN FILTER:')
predictions_svm_median = median_filter(15,'equalized_data')


print('CONTRAST DATA: ')
print('LPF:')
predictions_svm_lpf2 = gaussian_lpf(10,'contrast_data')
print('HPF:')
predictions_svm_hpf2 = gaussian_lpf(10,'contrast_data')
print('LPF+HPF:')
predictions_svm_lpf_hpf2 = gaussian_lpf_hpf(2,'contrast_data')
print('MEDIAN FILTER:')
predictions_svm_median2 = median_filter(3,'contrast_data')

all_pred = np.vstack((predictions_svm_lpf,predictions_svm_lpf2,predictions_svm_lpf_hpf,predictions_svm_median,predictions_svm_hpf2
                    ,predictions_svm_lpf2,predictions_svm_hpf2))

final_pred = np.zeros((len(predictions_svm_lpf),))
for i in range(final_pred.shape[0]):
    keys = Counter(all_pred[:,i]).keys()
    values = Counter(all_pred[:,i]).values()
    
    keys = np.array(list(keys))
    values = np.array(list(values))
    max_idx = np.argmax(values)
    final_pred[i] = keys[max_idx]

confusion(y_test,final_pred,'Final Confusion Matrix for Tumor Detection') 

print('FINAL ACCURACY OF THE ENSEMBLE CLASSIFIER IS: ' + str(accuracy_score(y_test,final_pred)))