import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from skimage.feature import hog

##
## feature engineering functions
##
def color_hist(img, nbins=32, bins_range=(0,256), debug = False) :
    """
    color_hist : takens in an mage and computes the histogram of the channels
    
    return: hist_feature_vec (vector of histograms of all 3 channels)
    """
    ch1 = np.histogram(img[:,:,0] , nbins, bins_range)
    ch2 = np.histogram(img[:,:,1] , nbins, bins_range)
    ch3 = np.histogram(img[:,:,2] , nbins, bins_range)
    
    hist_features_vec= np.concatenate((ch1[0],ch2[0],ch3[0]))
    
    if(debug == True) :
        bin_edges = ch1[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
        return hist_features_vec, ch1, ch2, ch3, bin_centers
    
    return hist_features_vec

def spatial_bin_features(img,size=(32,32)) :
    """
    spatial binning
    returns a spatially binned feature vector
    """
    features = cv2.resize(img, size).ravel()
    return features

def cvt_2_colorspace(rgb_img, cspace='RGB') :
    """
    Converts rgb_img to the colorspace defined int he 
    """
    if cspace != 'RGB':
        if cspace == 'HSV' :
            feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        elif cspace == 'HLS' :
            feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
        elif cspace == 'LUV' :
            feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LUV)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
        else :
            raise ValueError("Unknown cspace parameter - " + cspace)
    else :
        feature_image = np.copy(rgb_img)
    
    return feature_image

def get_hog_features(img, orient, pix_per_cell, cell_per_blk, transform_sqrt=True, vis=False, feature_vec = True) :
    """
    get_hog_features : returns a  vector of HOG (Histogram of Gradient) features
    If the vis argument is passed in with a value True then we return the hog_image along with the feature vector
    return : HOG features vector
    """
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_blk, cell_per_blk), transform_sqrt=transform_sqrt, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else :
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_blk, cell_per_blk), transform_sqrt=transform_sqrt, 
                            visualise=vis, feature_vector=feature_vec)
    return features

###
### Feature Extract methods 
###
def extract_features(img_files, cspace='RGB',include_color_features=False,
                     spatial_size=(32, 32),hist_bins=20, hist_range=(0, 1),
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, feature_vec=True):
    """
    extract_features
    Given a list of image file names, this method will load the image and gets the hog features for each image.
    If the include_color_features argument is True, then it will also include spatial binning and color histogram 
    features in to the features vector. By default the color features are not included
    
    return : feature vectors (numpy multi-dim array) of all images
    """
    features=[]
    for file in img_files: 
        img = mpimg.imread(file)
        features.append(single_img_extract_features(img,cspace,include_color_features,
                               spatial_size, hist_bins, hist_range,
                               orient, pix_per_cell, cell_per_block, hog_channel, feature_vec))
    return features

def single_img_extract_features(img, cspace='RGB',include_color_features=False,
                     spatial_size=(32, 32),hist_bins=20, hist_range=(0, 1),
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, feature_vec=True):
    """
    single_img_extract_features
    Given an image, extract features (HOG, [spatial, hist_bins]) and return a feature
    return : feature vector for the image
    """
    feature_image = cvt_2_colorspace(img, cspace=cspace)
    
    # 1. add hog features
    if hog_channel =='ALL' :
        hog_features = []
        for i in range(feature_image.shape[2]) :
            hog_features.append(
                get_hog_features(feature_image[:,:,i], 
                                 orient, pix_per_cell, cell_per_block, vis=False, feature_vec= feature_vec))
        hog_features = np.ravel(hog_features)
    else :
        hog_features = get_hog_features(feature_image[:,:,hog_channel], 
                                 orient, pix_per_cell, cell_per_block, vis=False, feature_vec= feature_vec)
    if include_color_features == True :
        #add spatial and color histogram features
        spatial_features = spatial_bin_features(feature_image, size=spatial_size)
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        
        return np.hstack((spatial_features, hist_features, hog_features))

    return hog_features
