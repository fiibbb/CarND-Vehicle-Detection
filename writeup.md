##Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/hog_example_0.jpg
[image3]: ./output_images/test6_pipeline.jpg

####Feature Extraction

The code for this step is contained in `VehicleDetection.train`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Here are some examples using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####HOG Parameters

After trial and error, I settled on the these parameters for HOG feature extraction: `color_space="YCrCb"`, `orientations=9`, `pixels_per_cell=8`, `cells_per_block=2`, as shown in `VehicleDetection.__init__`

####SVM Training

I trained using the above feature extraction, as shown in `VehicleDetection.train`, achieving and accuracy of around 98%.

####Sliding Window Search and HOG Sub-sampling

I then used sliding window technique and sub-sampling technique to analyze a single frame, as shown in `VehicleDetection.process_img`.  

The `find_boxes` function takes in an image, calculates the hog features of the entire image after croping and scaling, and generates a sequence of boxes. Each box is used to sub-sample the hog features. The extract sub-sample is then fed into the SVM for prediction. All boxes where the SVM predicts to be a car image is returned. Note that only a portion of the original image is processed (between `ystart` and `ystop`), and this portion is scaled to `scale` before hog features are calculated.  

The `find_boxes_with_scales` function generates a list of scales with their corresponding `(ystart, ystop)`, and calls `find_boxes` for each of the combination.  

The `draw_boxes_using_heatmap` function takes the boxese generated above, and use heatmap to remove false positives. The threshold is set to `VehicleDetection.threshold`. A pixel is only accounted for if there's number of boxes containing this pixel exceeds the threshold.  

Finally `process_img` keeps track of the boxes detected in the last `VehicleDetection.queue_frames` image frames. For every frame, the boxes of the current frame as well as boxes from last couple of frames are passed into `draw_boxes_using_heatmap`. Thus, we achive better false positive elimination and preventing the boxes from jumping around in the video.

An example of the pipeline shown as below:  

![alt text][image3]  

####Video
Here's a [link to my video result](./output.mp4)

As mentioned above, the video processing considers a series of frames to decide the bounding boxes, as opposed to considering each frame independently. For each frame, boxes detected in the last `VehicleDetection.queue_frames` frames are also added to the heatmap for labeling.


####Discussion of problems / issues

The overall result of this pipeline is accetable. It is able to detect cars in the test video most of the time. There are few cases where there are still false positives that made to the final video. This is a result of not sufficiently fine-tuning the threshold of heatmap. It turns out to rather painful tuning the combination of handling adjacent frames and thresholding heatmap, because the the more frames we consider, the higher the threshold need to be for compensating the added heatmap from recent frames. Another thing is that ideally the number of adjacent frames to consider should also depend on the speed of the vehicles -- the camera takes snapshots on a fixed interval, so the relative speed of the vehicles decides how far apart the the car image will jump in term of pixels between frames.
