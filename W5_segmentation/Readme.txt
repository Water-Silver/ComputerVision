Image Segmentation
====================

Usage Notes
------------

- Each source program intends to perform image segmentation.
- Following methods were used.
 
  Otsu's Method for bimodal segmentation on gray scale image. (otsu.cpp)
  Adaptive Thresholding Using Moving Averages on gray scale image (adaptivethreshold.cpp)
  K-Means Clustering using the opencv function for both gray and color image. (kmeans.cpp)


  
Requirements
-------------
Opencv should be installed and system environment variable should be added.
Relevant libraries should be imported.
The original image should be located in the project directory (writing.jpg, lena.jpg) 

