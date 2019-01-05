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

Mat gaussianfilterRGB_sep(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

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
	output = gaussianfilterRGB_sep(input, 1, 1, 1, "adjustkernel"); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Gaussian Filter RGB Sep", WINDOW_AUTOSIZE);
	imshow("Gaussian Filter RGB Sep", output);


	waitKey(0);

	return 0;
}


Mat gaussianfilterRGB_sep(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	
	int m = n;
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denomS;
	float denomT;
	float kernelvalue;

	// Initialiazing Kernel Matrix 
	float kernelS[3];
	float kernelT[3];


	denomS = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator s

		float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
		kernelS[a + n] = value1;
		denomS += value1;


	}
	denomT = 0.0;
	for (int b = -n; b <= n; b++) {
		float value2 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
		kernelT[b + n] = value2;
		denomT += value2;
	}


	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-paddle")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							kernelvalue = (kernelS[a + n] / denomS) * (kernelT[b + n] / denomT);
							sum1_r += kernelvalue*(float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += kernelvalue*(float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += kernelvalue*(float)(input.at<C>(i + a, j + b)[2]);
							
						}
					}
				}

				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;


			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						
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
						
						kernelvalue = (kernelS[a + n] / denomS) * (kernelT[b + n] / denomT);
						sum1_r += kernelvalue*(float)(input.at<C>(tempa,tempb)[0]);
						sum1_g += kernelvalue*(float)(input.at<C>(tempa,tempb)[1]);
						sum1_b += kernelvalue*(float)(input.at<C>(tempa,tempb)[2]);
					}
					output.at<C>(i, j)[0] = (G)sum1_r;
					output.at<C>(i, j)[1] = (G)sum1_g;
					output.at<C>(i, j)[2] = (G)sum1_b;
				}
			}
			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

					
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {


							kernelvalue = (kernelS[a + n] / denomS ) *(kernelT[b + n] / denomT);
							
							sum1_r += kernelvalue*(float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += kernelvalue*(float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += kernelvalue*(float)(input.at<C>(i + a, j + b)[2]);
							sum2 += kernelvalue;


						}
					}
				}

				output.at<C>(i, j)[0] = (G)(sum1_r / sum2);
				output.at<C>(i, j)[1] = (G)(sum1_g / sum2);
				output.at<C>(i, j)[2] = (G)(sum1_b / sum2);

			}
		}
	}
	return output;
}