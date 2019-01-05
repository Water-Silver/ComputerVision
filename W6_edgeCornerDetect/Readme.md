<h1>"Edge and Corner Detection"</h1>


<h2>Usage Notes</h2>


- The programs detect edges and corners using following methods repectively.

 * Canny.cpp uses Canny() in OpenCV

 * LoG.cpp uses Gaussian Filter for low pass filter and Laplacian Filter for

   high pass filter to obtain a clear image of edges. 
   
   
   ![log canny](https://user-images.githubusercontent.com/36324014/50730047-7dc79600-1187-11e9-8a3d-ab48538304d7.JPG)


 * Harris_corner.cpp uses cornerHarris() in OpenCV and NonMaximum_Suppression()

   cornerSubPix() optionally.
   
   ![harriscorner](https://user-images.githubusercontent.com/36324014/50730053-90da6600-1187-11e9-8df8-87e166cc4e94.JPG){width=50}


Requirements

-------------

Opencv should be installed and system environment variable should be added.

Relevant libraries should be imported.

The original image should be located in the project directory (writing.jpg, lena.jpg) 
