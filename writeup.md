**Vehicle Detection Project**

The goals/steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/sample_image.png
[image2]: ./output_images/hog_image1.png
[image3]: ./output_images/hog_image2.png
[image4]: ./output_images/windows_test5.jpg
[image17]: ./output_images/hot_windowstest5.jpg
[image18]: ./output_images/heatmap_test5.jpg
[image19]: ./output_images/final_test5.jpg

[image5]: ./video_output_images/heatmap_20170525-221334video.png
[image6]: ./video_output_images/heatmap_20170525-221337video.png
[image7]: ./video_output_images/heatmap_20170525-221340video.png
[image8]: ./video_output_images/heatmap_20170525-221343video.png
[image9]: ./video_output_images/heatmap_20170525-221346video.png
[image10]: ./video_output_images/heatmap_20170525-221350video.png

[image11]: ./video_output_images/windows_20170525-221333video.png
[image12]: ./video_output_images/windows_20170525-221336video.png
[image13]: ./video_output_images/windows_20170525-221339video.png
[image14]: ./video_output_images/windows_20170525-221342video.png
[image15]: ./video_output_images/windows_20170525-221345video.png
[image16]: ./video_output_images/windows_20170525-221349video.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 21 through 33 of the file called `feature_extractor.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` the output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

And here's another example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(10, 10)` and `cells_per_block=(4, 4)`:

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally used following settings:

~~~
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 2  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
~~~

The selection was based on the visual inspection of results and testing on test images and test video. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC and hinge loss function. This classifier used HOG, Histogram and spatial features for training. The reason Linear SVC was used is that according to the documentation, this function has a better underlying implementation. Here's what they have in the documentation:

*Similar to SVC with parameter kernel=’linear,' but implemented regarding liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.*

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used window sizes in range (64, 64), (96, 96), (128, 128), (192, 192). I also restricted searching in y-dimension but let it search in all of the x-dimension. The reason I choose these dimension is based on some manual approximationms of how big or small an image could appear in given frame. The overlap factor initially was 0.25 but that caused pipeline to be really slow. With some experimentation, I was able to increse it up to 0.75  that resulted in higher speed without much loss in precision.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I started with YUV color spaved but but eded up using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a great result.  Here are some example images:

![alt text][image4]
![alt text][image17]
![alt text][image18]
![alt text][image19]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here are the videos with color space YCrCb:

* [Project Video](./out_project_video.mp4)
* [Project Test Video](./out_test_video.mp4)

I also did another version of videos with color space YUV:

* [Project Video](./out_project_video1.mp4)
* [Project Test Video](./out_test_video1.mp4)

The second version shows that just changing color space results in some false positives. 

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Here are six frames and their corresponding heatmaps:

![alt text][image11]
![alt text][image5]

![alt text][image12]
![alt text][image6]

![alt text][image13]
![alt text][image7]

![alt text][image14]
![alt text][image8]

![alt text][image15]
![alt text][image9]

![alt text][image16]
![alt text][image10]

### Here the resulting bounding boxes are drawn onto the video file:

[Project Test Video](./out_test_video.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here are few challanges with this solution:

* This approach is very dependent on selected color space whihc may result in unexpected behaviour in very bright, dark or atmosphere that has hue of some different color.
* More data for training could help with robust model
* Similar to previous assignment, keeping track of continuous frame could help with more stable detection for videos.


