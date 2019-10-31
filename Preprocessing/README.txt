#################
#File definition
#################
EnhancementLib.py : contains all the enhancement function definition (eg: add_contrast())
Preproc.py : contains the main preprocessing code

#################
#Output images
#################
The operations are done in the following order. This means step N was obtained by transforming the output of step N-1 :
1) original_data : images of different size and bit depth
2) grayscale_data : All in 8-bit grayscale
3) resized_data : All of the same size (h350*w300)
4) contrasted_data : Applied 1/(1+exp(-0.02(x-127))) contrast function
5) equalized_data : Applied histogram equalization
