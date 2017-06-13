
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/RGB_histogram.png
[image4]: ./examples/bboxes.png
[image5]: ./examples/labels_map.png
[image6]: ./examples/output_bboxes.png
[image7]: ./examples/output_bboxes_and_heatmap1.png
[image8]: ./examples/output_bboxes_and_heatmap2.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.    

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting HOG features is contained in the cells 13 to 16 of the IPython notebook (VehicleDetection.ipynb) and also in file  `features.py`(used this from car_classifier.py which I needed for training the classifier outside Ipython as it was crashing sometimes).  

I started by reading in all the `vehicle` and `non-vehicle` images (cells 2 to 5).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored RGB histogram and spatial binning features of both cars and non-cars (cells 6-12) . See screenshot of histogram features

![alt text][image3]

As shown in the screenshot, histogram does show some differentiation between the 2 image types. We will use color histogram with other complex features in our classifier. 

I then explored Histogram of Gradients (HOG) features with different color using `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and with just R, G or B channel and also including all channels. Initially I used just 1 channel but the classifier that I trained with this was only getting 94% accuracy. Then when I used all channels I could get a higher accuracy(98.4) though the feature size became larger(5292). I finally settled with orientation =10, pixels_per_cell=(8,8) and cells_per_block=(2,2). I ran the HOG features on all 3 channels with different color spaces too. I also identified parameter for color_hist( hist_bins = 16 & bins_range = (0,256)), color_space='YCrCb'  and spatial_binning features(    spatial_size=(16,16))

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG features, color hist features and spatial binning features (feature extraction code is in features.py). The code for training the classifier is in car_classifier.py. I tried using an RBF kernel. I trained the SVM with a 3-fold crossvalidator and did a grid search to find the right svm_C and svm_gamma(for RBF kernel). The RBF kernel was very slow to train and predict. So I reverted back to linear SVM which was getting me 98.4% accuracy with some hard negative mining as compared to 99.1% accuracy with a very slow RBF kernel.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search code is in cells 25 -27. In cell 25 I have defined a find_cars method that essentially uses a sliding window to scan the image in small sections, compute their features for the window and then use a classifier to identify if the section of image is a car or non-car. Computing hog everytime is an expensive operation. So to optimize performance, we will compute the HOG of the entire image once and then use the window to step through hog feature for that window section. The downside is that it is an approximation of the HOG if you had computed it individually and so does result in more false positives. The find_cars method also takes in a y_start and y_stop parameter because for our problem where the camera shows images of driving in lanes, cars will be found only below the horizon(around y = 400). So we can restrict the one time HOG to just the ystart and y_stop sections. Also we use a scale parameter to enable us to scan the images with different window sizes. The trick we do in this method is that instead of changing the window size, we just resize the image for the scale parameter so that when you scan the image with the same window size, the scaling effectively results in as though u were using different window size. e.g., say you scan a 400x400 image with a window size 64. Now if you resize the same image to 200x200 but the window size is still the same, your effective scanning window sie would be 128.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales(1.0, 1.5 and 2.0) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code to solve for false positive and duplicate bounding boxes are in cells 28-35. I get the heatmaps of the cars by calling the the find_cars method with 3 scales and then using a heatmap threshold remove out false positives. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap each corresponding to a vehicle.  Now using these label position draw bounding boxes on the original image.

### Here is the output of `scipy.ndimage.measurements.label()` on the one of the images:
![alt text][image5]

Here's an example result showing the heatmap for the test images and the computed bounding boxes :

### Here are six frames with their heatmaps and corresponding bounding_boxes:

![alt text][image7]
![alt text][image8]


### Smoothing the bounding boxes across frames 
I created a Tracker class that tracks the heatmaps of cars across frames. The code for the Tracker class and how its called from the pipeline are in cells 36 - 38. To smooth the frames, we aggregate the heatmaps of images across n frames. On the nth frame we will use the heatmap threshold and `scipy.ndimage.measurements.label()` to compute the new label which is stored in the last_label attribute in the class. We then reset the agg_hashmap for the next n-frame aggregation. For the next n frames the bounding boxes are drawn from the last_label value while the agg_hashmap accumulates the heat map. In this way we smooth the bounding boxes across frames. This also cuts down false positives as most false positives do not carry over across frames. So when aggregating across frames these false positives will have lower heatmap values and would get trimmed out by the threshold.

### Project Output Video

![alt text][video1]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Had to do some hard negative mining to remove false positives in the classifier. So the classifer would be overfit to the problem and may not generalize well. Tried to use RBF Kernel but it was slower. Also computing HOG features is expensive and slower. It takes about 20 minutes to detect the cars in teh project video. Need a faster model to work in realtime. 

I would try object detection neural net architectures like YOLO - https://pjreddie.com/media/files/papers/yolo.pdf that uses a convolutional neural net architecture to predict bounding boxes of the objects that we want to detect.

