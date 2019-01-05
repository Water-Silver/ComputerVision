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

Mat laplacianfilter_rgb(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);
	output = laplacianfilter_rgb(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", output);


	waitKey(0);

	return 0;
}


Mat laplacianfilter_rgb(const Mat input) {

	//Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

			   // Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
			   //Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
	float Ldata[] = { 0,1,0,1,-4,1,0,1,0 };
	

	Mat L(3, 3, CV_32F, Ldata);
	int tempa;
	int tempb;
	

	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
			float sum_r = 0.0;
			float sum_g = 0.0;
			float sum_b = 0.0;

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
					sum_r += L.at<float>(a + n, b + n)*(float)(input.at<C>(tempa, tempb)[0]);
					sum_g += L.at<float>(a + n, b + n)*(float)(input.at<C>(tempa, tempb)[1]);
					sum_b += L.at<float>(a + n, b + n)*(float)(input.at<C>(tempa, tempb)[2]);
					

				}
			}
			
			float max = sum_r;
			if (sum_g > max) max = sum_g;
			if (sum_b  > max) max = sum_b;
			//sum1 = (sum_r + sum_g + sum_b) / 3.0;
			//if (sum1 < 0)sum1 = 0;
			//if (sum1 > 255) sum1 = 255;
			if (max < 0)max = 0;
			if (max > 255) max = 255;
			output.at<C>(i, j)[0] = (G)(max);
			output.at<C>(i, j)[1] = (G)(max);
			output.at<C>(i, j)[2] = (G)(max);

		
		}

	}

	return output;


}
