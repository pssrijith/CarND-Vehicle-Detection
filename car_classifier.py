import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import features as ft
import pickle

### the training pipeline method
def train_svm_pipeline(X_train, y_train) :
    """
    train a svm pipeline with RBF kernel using a 5 fold cross validator. The hyperparams will be gridsearched
    using the cross validator
    """
    # Check the training time for the SVC
    t=time.time()
    # Use a standard Scaler
    scaler = StandardScaler()
    
    # Use SVC with kernel
    svc = LinearSVC()
    
     # Create Pipeline with components [scaler, svc]
    svm_pipeline = Pipeline([('scaler', scaler), ('svc', svc)])
    
    # Create stratified k fold cross validator
    k_fold=3
    cv = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2, random_state=42)
    
    # set pipeline params(classifier params, scaler params) int hte   for Grid Search 
    C_range = np.logspace(-1, 1, 5)
    #gamma_range = np.logspace(-5, 1, 5)
    #C_range=[1.0]
    #gamma_range=[ 0.0001]
    # pipeline component parameters should be prefixed with the component name 
    # e.g., 'svc' params will have the prefix svc__
    #param_grid = dict(svc__C=C_range, svc__gamma=gamma_range)
    param_grid = dict(svc__C=C_range)
    # grid search pipeline with CV and the pipeline params
    svm_grid_pipeline = GridSearchCV(svm_pipeline, param_grid=param_grid, cv=cv)
    print("Starting Grid Search...")
    svm_grid_pipeline.fit(X_train, y_train)
    print("The best parameters are %s with a score of %0.2f"
      % (svm_grid_pipeline.best_params_, svm_grid_pipeline.best_score_))
    
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    
    return svm_grid_pipeline

#########################################################################
# Main logic 
# load the car and non-car images, train the clasifier and save the model
#########################################################################
def main() :
    cars = glob.glob("./train_data/vehicles/*/*.*")
    print("cars input size :",len(cars))
    non_cars = glob.glob("./train_data/non-vehicles/*/*.*")
    print("non-cars input size :", len(non_cars))

    #sample_size = 4000
    #cars= cars[0:sample_size]
    #non_cars= non_cars[0:sample_size]
    #print("cars sample size",len(cars))
    
    #### Set feature extraction parameters
    # colorspace
    cspace = 'YCrCb'
    # spatial bin param
    spatial_size=(16,16)
    #hist params
    hist_bins = 16
    bins_range = (0,256)
    #hog params
    orient =10
    pix_per_cell = 8
    cells_per_block = 2
    hog_channel = 'ALL'
    include_color_features=True
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cells_per_block,'cells per block')
    
    ### Extract Features
    t1=time.time()
    car_features = ft.extract_features(cars, cspace, include_color_features,
                         spatial_size,hist_bins, bins_range,
                         orient= orient, pix_per_cell = pix_per_cell, cell_per_block= cells_per_block, 
                         hog_channel = hog_channel)
    non_car_features = ft.extract_features(non_cars, cspace, include_color_features, 
                         spatial_size,hist_bins, bins_range,
                         orient= orient, pix_per_cell = pix_per_cell, cell_per_block= cells_per_block, 
                         hog_channel = hog_channel)
    t2=time.time()

    print("Time taken to extract images %0.2fs"%(t2-t1))

    X= np.vstack((car_features,non_car_features))
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
    
    (X,y) = shuffle(X,y, random_state=42)
    ## create train and test
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print("X Train shape",X_train.shape)
    print("y Train shape",y_train.shape)
    print("X Test shape",X_test.shape)
    print("y Test shape",y_test.shape)
    
    ### Train SVM pipeline 
    svm_pipeline_model = train_svm_pipeline(X_train,y_train)

    ### Verify test scores on trained model
        # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svm_pipeline_model.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    n_predict = 10
    t=time.time()
    print('My SVC predicts: ', svm_pipeline_model.predict(X_test[0:n_predict]))
    t2 = time.time()
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    # save model and feature params used for extraction
    dist_features = {'svm_pipeline_model':svm_pipeline_model,
                     'cspace':cspace,
                     'orient':orient,
                     'pix_per_cell':pix_per_cell,
                     'cells_per_block':cells_per_block,
                     'hog_channel':hog_channel,
                     'spatial_size':spatial_size,
                     'hist_bins':hist_bins,
                     'bins_range':bins_range,
                     'include_color_features':include_color_features
                    }
    print("saving model...")
    pickle.dump(dist_features,open( "./svm_model.p", "wb" ))

if __name__ == '__main__' :
    main()