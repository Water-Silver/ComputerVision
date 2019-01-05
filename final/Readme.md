<h1>Final Project<h1>

<h2>Automatic Document Scanning</h2> 

Fixes projection distortion of a picture of a document and saves the result image with better quality.
Finding corners, applying image warping, and adjusting contrasts were done. (C++, OpenCV)

The Process is as followed.
1. Apply K-means clustering and Mean filtering to blur out the contents on the paper.

K-means

![1 kmeans](https://user-images.githubusercontent.com/36324014/50730213-5ffc3000-118b-11e9-9166-7e400889c5ad.PNG)

mean filtering

![2 meanfiltered](https://user-images.githubusercontent.com/36324014/50730228-efa1de80-118b-11e9-9b8b-137c1abe720b.PNG)

2. Then use cornerHarris function provided by OpenCV to find corners of the paper.

![4 harriscorner](https://user-images.githubusercontent.com/36324014/50730226-e44eb300-118b-11e9-9fbf-516776bf9884.png)

3. Undistort the image by applying perspective transformation using the corners found.

4. Adjust contrast using Adaptive Threshold or Unsharp Masking.

Result

![7 final_image_grayscale](https://user-images.githubusercontent.com/36324014/50730263-89698b80-118c-11e9-96b2-29a2b0c72f49.png)
