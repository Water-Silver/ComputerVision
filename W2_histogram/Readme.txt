Histogram Equalization and Sepcification on both gray and color image.
=====================================================================

Usage Notes
------------

- The program intends to alter the histogram of an image and show the equalized/matched image.

- Following functions are defined in the hist_func.h file.

  float *cal_PDF(Mat &input)        // generate PDF for single channel image
  float **cal_PDF_RGB(Mat &input)   // generate PDF for color image
  float *cal_CDF(Mat &input)        // generate CDF for single channel imgae
  float **cal _CDF_RBG(Mat &input)  // generate CDF for color image
  
- Each source file generates :
 
  hist_stretching.cpp : stretched image and txt files of histogram of original and stretched image.
  hist_eq.cpp : equalized image, txt files of 1 transfer function and histogram of original and output image each.
  hist_eq_RGB.cpp : equalized image, txt files of 3 transfer function and 3 histograms of original and output image respectively.
  hist_eq_YUV.cpp : equalized image, txt files of 1 transfer function for channel Y, and 3 histogram of the original and output image respectively.
  hist_match.cpp : histogram matched image, txt files of 1 transfer function, histogram of original and output image each.
  hist_match_YUV.cpp : histogram matched image, txt files of 1 transfer function for Y channel, and 3 histogram of the original and output image respectively.
 
  
Requirements
-------------
Opencv should be installed and system environment variable should be added.
Relevant libraries should be imported as well.
The original image should be located in the project directory