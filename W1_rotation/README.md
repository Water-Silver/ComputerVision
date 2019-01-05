
<h1>"Image Rotation using nearest and bilinear interpolation"</h1>

![rotationresult](https://user-images.githubusercontent.com/36324014/50729872-e5c8ad00-1184-11e9-8d03-2dcd9cf1a9b1.JPG)

<h2>Usage Notes</h2>


- The program is intened to rotate a given image and show it on screen.

- 

  for (int i = 0; i &lt; sq_row; i++) { 

       for (int j = 0; j &lt; sq_col; j++) {

float x = (j - sq_col / 2) * cos(radian) + (i - sq_row / 2) * sin(radian) + col / 2;

float y = - (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

These lines are converting the pixels in the original image into 

the pixels in the newly made rotated image.

- By "nearest" the floor() function will be used to fill the pixel with the closest pixel's intensity.

- By "bilinear" the pixel's intensity will by calculated by the surounding 4 points.

  The way to do this is : calculate the intensity of points between the two upper points(f(x',y))

  and between the two bottom points (f(x+1,y)) respectively.

  Then calculate the intensity of f(x',y')= ¥ëf(x+1,y') + (1-¥ë)f(x,y')  

<h2>Requirements</h2>

Opencv should be installed and system environment variable should be added.

Relevant libraries should be imported as well.

The original image should be located in the project directory



