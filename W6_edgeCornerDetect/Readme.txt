Edge and Corner Detection
====================

Usage Notes
------------

- The programs detect edges and corners using following methods repectively.
 
 * Canny.cpp uses Canny() in OpenCV
 * LoG.cpp uses Gaussian Filter for low pass filter and Laplacian Filter for
   high pass filter to obtain a clear image of edges. 
 * Harris_corner.cpp uses cornerHarris() in OpenCV and NonMaximum_Suppression()
   cornerSubPix() optionally.

Requirements
-------------
Opencv should be installed and system environment variable should be added.
Relevant libraries should be imported.
The original image should be located in the project directory (writing.jpg, lena.jpg) 

