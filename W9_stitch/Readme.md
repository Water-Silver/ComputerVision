<h1>"Fitting"</h1>


<h2>Usage Notes</h2>

- The programs uses corresponding points found by SIFT descriptor to perform affine transformation and image stitching.
- StitichingAffine.cpp generates the affine transform matrix using all matched points.
- StitchingWRansac.cpp generates the transformation matrix using RANSAC approach which
  compares the quality of transformation done by three different sets each consisting of three points(minimum number of points required
  to perform affine transform).


![matching ransac](https://user-images.githubusercontent.com/36324014/50730156-8325e000-1189-11e9-9b85-35e07b943681.JPG)

<h2>Requirements</h2>

Opencv ver 2 should be installed and system environment variable should be added.
Relevant libraries should be imported.
The original image should be located in the project directory.

