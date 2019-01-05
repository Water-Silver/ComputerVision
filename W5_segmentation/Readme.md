Image Segmentation
====================

Usage Notes
=========================
- Each source program intends to perform image segmentation.

- Following methods were used.

  Otsu's Method for bimodal segmentation on gray scale image. (otsu.cpp)
  
  ![otsuresult](https://user-images.githubusercontent.com/36324014/50730026-36d9a080-1187-11e9-8fda-27430b8e4689.JPG)


  Adaptive Thresholding Using Moving Averages on gray scale image (adaptivethreshold.cpp)
 
  ![adaptivethresholdresult](https://user-images.githubusercontent.com/36324014/50730028-42c56280-1187-11e9-97e1-a7e2319302c8.JPG)


  K-Means Clustering using the opencv function for both gray and color image. (kmeans.cpp)


Requirements
======================================

Opencv should be installed and system environment variable should be added.

Relevant libraries should be imported.

The original image should be located in the project directory (writing.jpg, lena.jpg) 
