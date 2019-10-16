# DIP-Tumor_Classification
A Brain Tumor Classification Project for Digital Image Processing class (ECE 6258) at Georgia Tech.

The file main_classifier.py and all others are pretty much the same except I add Low-Pass filter or HPF to the images. 
Here are the descriptions of files we have: 

original.py --> Read the images as it is, and make sure they are grayscale 
gaussian_lpf.py --> Read the images and apply gaussian low-pass filter 
gaussian_hpf.py --> Read the images and apply gaussian high-pass filter
gaussian_lpf_hpf.py --> Read the images and apply gaussian LPF and gaussian HPF after that. 

* The codes I wrote asks a folder name for input. Right now, we have two folders, resized_data and contrast_data, so you can just put one of those. 

RUNNING THE EXPERIMENTS: 

```
python3 original.py
resized_data
```


```
python3 gaussian_hpf_lpf.py
resized_data
```

