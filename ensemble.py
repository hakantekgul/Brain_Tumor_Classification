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


print('######## Starting Experiments ###############')
print('Please enter the folder name to read the images from:')
folder_name = input()

print('ORIGINAL:')
predictions_nn, predictions_nn2, predictions_svm, predictions_svm2, predictions_knn, y_test = original(0,folder_name)
print('LPF:')
predictions_nn_lpf, predictions_nn2_lpf, predictions_svm_lpf, predictions_svm2_lpf, predictions_knn_lpf = gaussian_lpf(5,folder_name)
print('HPF:')
predictions_nn_hpf, predictions_nn2_hpf, predictions_svm_hpf, predictions_svm2_hpf, predictions_knn_hpf = gaussian_hpf(5,folder_name)
print('LPF+HPF:')
predictions_nn_lpf_hpf, predictions_nn2_lpf_hpf, predictions_svm_lpf_hpf, predictions_svm2_lpf_hpf, predictions_knn_lpf_hpf = gaussian_lpf_hpf(5,folder_name)
print('Median:')
predictions_nn_median, predictions_nn2_median, predictions_svm_median, predictions_svm2_median, predictions_knn_median = median_filter(5,folder_name)


all_pred = np.vstack((predictions_nn, predictions_nn2, predictions_svm, predictions_svm2, predictions_knn
					,predictions_nn_lpf, predictions_nn2_lpf, predictions_svm_lpf, predictions_svm2_lpf 
					,predictions_knn_lpf,predictions_nn_hpf, predictions_nn2_hpf, predictions_svm_hpf 
					,predictions_svm2_hpf, predictions_knn_hpf, predictions_nn_lpf_hpf, predictions_nn2_lpf_hpf
					,predictions_svm_lpf_hpf, predictions_svm2_lpf_hpf, predictions_knn_lpf_hpf
					,predictions_nn_median, predictions_nn2_median, predictions_svm_median, predictions_svm2_median, predictions_knn_median))

final_pred = np.zeros((len(predictions_nn), ))
for i in range(final_pred.shape[0]):
    keys = Counter(all_pred[:,i]).keys()
    values = Counter(all_pred[:,i]).values()
    
    keys = np.array(list(keys))
    values = np.array(list(values))
    max_idx = np.argmax(values)
    final_pred[i] = keys[max_idx]

print(accuracy_score(y_test,final_pred))