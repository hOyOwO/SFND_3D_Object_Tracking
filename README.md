# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
  * Install Git LFS before cloning this Repo.
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.


## Final Project

1. FP.1: Match 3D objects
2. FP.2: Compute Lidar-based TTC
    * reference: Lesson 3
    * TTC = minXCurr * dT / (minXPrev - minXCurr)
3. FP.3 : Associate Keypoint Correspondences with Bounding Boxes
    * check current point is in bounding box or not
    * if inside bounding box, add keypoint and match
4. FP.4 : Compute Camera-based TTC
    * Make the ratio the distances each frame. (distCurr/ distPrev)
    * Calculate the TTC with median of ratio.
5. FP.5 : Performance Evaluation 1
    * Look for several examples where you have the impression that the Lidar-based TTC estimate is way off
    * provide a sound argumentation why you think this happened  
    * Good case  
    <img src = "refdata/example_images/goodCase1.png" width = 300>
    <img src = "refdata/example_images/goodCase2.png" width = 300>
    
    * Bad case  
    <img src = "refdata/example_images/badCase0.png" width = 300>
    <img src = "refdata/example_images/badCase1.png" width = 300>
    <img src = "refdata/example_images/badCase2.png" width = 300>

    * To terminate outlier, I used Normal Distribution method. see below result  
    <img src = "refdata/example_images/NormalDistributionGraph.jpg" width = 200>
    <img src = "refdata/example_images/ttcDiff.png" width = 200>
  6. FP.6 : Performance Evaluation 2
     * Compare lidar ttc and camera ttc (all variation)
     * Rank performance
     #### 1st Rank. Detector: AKAZE, Descriptor: SIFT   
     #### 2nd Rank. Detector: AKAZE, Descriptor: AKAZE
     #### 3rd Rank. Detector: SHITOMASI, Descriptor: BRIEF
     * See below table (-inf or nan occur, I assigned 100 to calculate)

<img src = "refdata/example_images/finalResult.png">

