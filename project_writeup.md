## Advanced Lane Finding

### This write-up documents the solution strategy and implementation for the Advanced Lane Finding Project of the Udacity Self-Driving Car Engineer Nanodegree Program course.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_image_with_corners.jpg "Found corners"
[image2]: ./output_images/cal_image_undistorted.jpg "undistorted calibration image"
[image3]: ./output_images/undistort_chessboard.jpg "Road Transformed"
[image4]: ./output_images/test_image_undistorted.jpg "Binary Example"
[image5]: ./output_images/straight_lane_warp_test.jpg "Warp Example"
[image6]: ./output_images/curved_lane_warp_test.jpg "Warp Example"
[image7]: ./output_images/abs_soble_thresh_test.jpg "Absolut Sobel Threshold"
[image8]: ./output_images/mag_thresh_test.jpg "Magnitude Threshold"
[image9]: ./output_images/hls_binary_test.jpg "HLS Select"
[image10]: ./output_images/color_pipeline.jpg "Color transform for pipeline"
[image11]: ./output_images/binary_warped_example.jpg "Binary Wapred example"
[image12]: ./output_images/warp_unwarp_example.jpg "Binary Unwapred example"
[image13]: ./output_images/histogram_visualization.jpg "Histogram Visualization"
[image14]: ./output_images/sliding_window_test_3.jpg "Lane finding Test Image 3"
[image15]: ./output_images/lane_detection_output_001.png "Lane overlay example"
[image16]: ./output_video/images/vlcsnap-2018-06-17-22h59m28s954.png "Failure example 1"
[image17]: ./output_video/images/vlcsnap-2018-06-17-17h40m07s385.png "Failure example 2"

[image18]: ./output_images/sliding_window_test_5.jpg "Lane finding failure"
[image19]: ./output_images/color_pipeline.jpg "Color transform for pipeline"

[video1]: ./output_video/project_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### The specific points of the rubric are considered and described individually, with explanation of my strategy and implementation.  

---

### 1. Camera Calibration

#### 1.1 Camera calibration was implemented using the provided chessboard calibration patterns and OpenCV functions.

The code for this step is contained in cell [5] of the IPython notebook, Advanced_Lane_Finding.ipynb.  

First `object points` are defined, which will represent the (x, y, z) coordinates of the chessboard corners in the real world, although it is assumed that the chessboard is fixed on the (x, y) plane at z=0, and that the object points are therefore the same for each calibration image.  

The `objp` array defines a grid with the same grid number as the calibration images. Two lists are define:

* `objpoints` = []
* `imgpoints` = []

 `objpoints` is appended with a copy for each successful detection of all the chessboard corners in a test image.  `imgpoints` is appended with the (x, y) pixel positions of each of the corners in the image plane.

`objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. This function iterated through the calibration images in the ./camera_cal/ folder and a final pickle file was written out containing the calibration data:

./camera_cal/calibration_pickle.p

Below is an example of the chessboard calibration image and with found corners:
![alt text][image1]

A distortion function was defined, which would use the cv2.undistort() function, referencing the calibration file data and "undistort" an image. An example of the undistortion applied to a calibration image is shown below:
![alt text][image2]

As part of the lane finding pipeline, a top-down view will be applied before performing lane finding. In cells [7] and [8] the undistort and top_down perspective transformation were applied to a chessboard calibration image. The top_down transformation is done using the cv2.warpPerspective() function. A straight line pattern is seen in the transformed image as shown below:
![alt text][image3]

### 2. Pipeline Transformations

#### 2.1 Provide an example of a distortion-corrected image.

The undistortion function [9] is applied to one of the test images below:

![alt text][image4]

#### 2.2 Top_Down Perspective Transform

The top_down perspective transform was applied using the `cv2.warpPerspective()` and `cv2.getPerspectiveTransform()` functions. The arguments for `cv2.getPerspectiveTransform()` include src and dst, which are the source and destination coordinates warpPerspectivly, defining how the image needs to be warped to achieve the top_down perspective. M and Minv then are referenced in `cv2.warpPerspective()` as needed for the top_down and then later the inverse transformation, which will be needed at the end of the pipeline. The `src` and `dst` coordinates are define as follows:

* bot_width = 0.74 # percent of bottom trapizoid height
* mid_width = 0.08 # percent of middle trapizoid height
* height_pct = 0.61 # percent for trapizoid width
* bottom_trim = 0.95 # percent from top to bottom to avoid car hood

* src = np.float32([[img.shape[1] * (0.5 - mid_width / 2), img.shape[0] * height_pct],
* [img.shape[1] * (0.5 + mid_width / 2), img.shape[0] * height_pct],
* [img.shape[1] * (0.5 + bot_width / 2), img.shape[0] * bottom_trim],
* [img.shape[1] * (0.5 - bot_width / 2), img.shape[0] * bottom_trim]])

* offset = img_size[0] * 0.25
* dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

This resulted in the following source and destination points, based on the image input dimensions of 1280 x 720 pixels:

| Source        | Destination   |
|:-------------:|:-------------:|
| 588.8, 439.2  | 320, 0        |
| 691.2, 439.2  | 960, 0        |
| 1113.6, 684   | 960, 720      |
| 166.4, 684    | 320, 720      |

This perspective transform [16] was place in the function `topdown()`, with the input image as the only function input. Examples of the topdown perspective transform on straight and curved lane examples are shown below, and in both cases the transformed lines appear parallel to one another along the lane length:

![alt text][image5]
![alt text][image6]

#### 2.3 Color Transforms Design and Testing.

Various color transform approaches were investigated, including:

* Sobel thresholds
* HLS selection
* Combined HLS and HSV

Absolute Sobel Threshold

![alt text][image7]

Magnitude Sobel Threshold
![alt text][image8]

HLS Select
![alt text][image9]

In the final binary transform function implemented in the lane finding pipeline, the `pipeline_color()` was defined including the Sobel X derivative, and S Channel threshold. The combined binary image is shown below after transformation:

![alt text][image10]

#### 2.4 Binary_Warped Output

The color and top_down transformations were combined before performing the lane finding and metric characterization, an example is shown below:
![alt text][image11]

The transformation from warp to unwarp (inverse topdown warp) was tested.

![alt text][image12]

### 3. Lane Finding and Characterization

#### 3.1 Lane Finding

Once the binary_wapred image is created, it can be analyzed using a histogram and sliding window approach to identify and track the path of a lane in a single image. Then a polynomial can be fit to the found pixel data corresponding to a lane.

![alt text][image13]

In the sliding windows approach, a window slides across the image, and looks for areas where the pixel values are not equal to zero. These pixel positions can then  be collected in a list, and the list appended with each new found section for the right and left lanes. To capture the lane curvature, a 2nd order polynomial can then be fitted to the lane curve segments, the curvature can then be calculated from the polynomial function.

The function sandbox(binary_wapred) was defined to visualize lane finding using the sliding window approach and test on different test images. This included the sliding window as well as the plotted line over the binary_wapred input image.

![alt text][image14]

For the full pipeline, sliding_windows(binary_wapred) function was defined for use in the final image processing pipeline. The size and number of windows is defined, and as pixel data is found, it is stored in one of two lists: left_lane_inds or right_lane_inds. From the pixel position data, a conversion was then made to the real world dimensions of the image (ym_per_pix and xm_per_pix).

#### 5. Draw Lane Function

Once the lane finding function was run to return the data for left_fit, right_fit and curve radius data, the draw_lane() function defined to draw the lane shape and lane metric data.

A key part of draw_lane is to warp the lane data from the topdown warped image to the unwarped state, which would then be annotated to the unwarped video image. This was accomplished by defining Minv, which is the inverse of the perspecitve warp function. Additionally, the lane curvature and vehicle position data was written using the cv2.putText function.

![alt text][image15]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Lane Finding Failures

In the pipeline development, different approaches to the color transformation were attempted, and tested on the video pipeline. There were some specific road sections which were problematic for lane detection. Generally speaking, in transition regions where the lane material changed color, there could be lane detection problems, resulting in the lane moving too far into the shoulder of the lane.

The threshold definitions for the color transformation played a large role in land finding failure. The final values for the pipeline_color() function were:
* s_thresh = (115, 200)
* sx_thresh = (20, 150)

Am extreme failure example with non-ideal thresholds is shown below, where the lane is completely mismatched from the green overlay shape.
![alt text][image16]

A more common failure is shown below, where the left lane was not found correctly. This was likely related to the features of the lane shoulder, where the binary_warped image would give the impression that the lane was far to the left. This would be corrected farther down the highway, but presented an issue with successfully segmenting the lane features.
![alt text][image17]

An example of the left lane finding failure is shown below, where the high density of white from the lane shoulder produced an incorrect result from the algorithm.

![alt text][image18]

The lane finding failure may be improved by masking the search region of the image, similar to the initial lane finding project. This could prevent the sliding window from using the binary color information from that section of the lane.
