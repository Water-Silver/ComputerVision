/*code was given*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat sobelfilter(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;


	cvtColor(input, input_gray, CV_RGB2GRAY);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	/*
	float Sxdata[] = {  -1,0,2 ,-2,0,2 , -1,0,1  };
	float Sydata[] = { -1,-2,-1 , 0,0,0, 1,2,1 };
	Mat Sx(3, 3, CV_32F, Sxdata);
	Mat Sy(3, 3, CV_32F, Sydata);
	//std::cout << "Sy=" << Sy << std::endl;
	*/
	

	//namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	output = sobelfilter(input_gray); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);


	waitKey(0);

	return 0;
}


Mat sobelfilter(const Mat input) {

	//Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
	float Sxdata[]= { -1,0,1,-2,0,2,-1,0,1 };
	float Sydata[] = { -1,-2,-1,0,0,0,1,2,1 };
	
	Mat Sx(3, 3, CV_32F, Sxdata);
	Mat Sy(3, 3, CV_32F, Sydata);
	int tempa;
	int tempb;
	
	
	Mat output = Mat::zeros(row, col, input.type());
	
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 

				           if (i + a > row - 1) {  //mirroring for the border pixels
								tempa = i - a;
							}
							else if (i + a < 0) {
								tempa = -(i + a);
							}
							else { // in the boundary
								tempa = i + a;
							}
							if (j + b > col - 1) {
								tempb = j - b;
							}
							else if (j + b < 0) {
								tempb = -(j + b);
							}
							else {
								tempb = j + b;
							}
							sum1 += (float)input.at<G>(tempa, tempb)*Sx.at<float>(a + 1, b + 1);
						    sum2 += ((float)input.at<G>(tempa, tempb)*Sy.at<float>(a+1, b+1));
							
						}
					}
			        float sum3 = pow(sum1, 2) + pow(sum2, 2);
			      
			         sum3 = sqrt(sum3);
					
					
			        output.at<G>(i, j) = (G)(sum3);
				}
			 
			}
		
	return output;
	
	
}
