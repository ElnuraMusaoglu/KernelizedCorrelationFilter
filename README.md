# KernelizedCorrelationFilter

 High-Speed Tracking with Kernelized Correlation Filters

  J. F. Henriques   R. Caseiro   P. Martins   J. Batista
                   TPAMI 2014
                   
 This is a Python implementation of High-Speed Tracking with Kernelized Correlation Filters
 
 
References

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista, "High-Speed Tracking with
Kernelized Correlation Filters", TPAMI 2014 (to be published).

[2] Y. Wu, J. Lim, M.-H. Yang, "Online Object Tracking: A Benchmark", CVPR 2013.
Website: http://visual-tracking.net/

[3] P. Dollar, "Piotr's Image and Video Matlab Toolbox (PMT)".
Website: http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html

[4] https://github.com/lawpdas/fhog-python

# The steps to use Kernelized Correlation Filter using HOG and VGG-19 deep features

## 1. Install and Build fhog function of PDollar Toolbox to the folder hog_cpp

python setup.py build_ext --inplace

## 2. Download the VGG-Net-19 model, replace to model folder
##    Set vgg_path in kernelized_correlation_filter.py line 15

http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

# 3. Download OTB100 Dataset to the dataset folder

http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html
