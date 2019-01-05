#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <stdio.h>
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

Mat gaussianfilter_sep(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {
	
	LARGE_INTEGER liCounter1, liCounter2, liFrequency;
	QueryPerformanceFrequency(&liFrequency); // 주파수(1초당 증가되는 카운트수)를 구한다.
	QueryPerformanceCounter(&liCounter1); // 코드 수행 전 카운트 저장
	
	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;


	cvtColor(input, input_gray, CV_RGB2GRAY);



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	output = gaussianfilter_sep(input_gray, 1, 1, 1, "adjustkernel"); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Gaussian Filter Sep", WINDOW_AUTOSIZE);
	imshow("Gaussian Filter Sep", output);
	QueryPerformanceCounter(&liCounter2);
	printf("수행시간 = %f 초 \n", (double)(liCounter2.QuadPart - liCounter1.QuadPart) / (double)liFrequency.QuadPart);
	
	waitKey(0);

	

	
	return 0;
}


Mat gaussianfilter_sep(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel;
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
		
			float value1 =exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
			kernelS[a+n]= value1;
			denomS+= value1;
			
		
	}
denomT = 0.0;
		for (int b = -n; b <= n; b++) {
			float value2 = exp( - (pow(b, 2) / (2 * pow(sigmaT, 2))));
			kernelT[b+n] = value2;
			denomT += value2;
		}

	
	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-paddle")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							kernelvalue = kernelS[a+n]/denomS;
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b))*(kernelT[b+n]/denomT);
						}
					}
				}

				output.at<G>(i, j) = (G)sum1;

			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
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
						kernelvalue = kernelS[a + n] / denomS;
						sum1 += kernelvalue*(float)(input.at<G>(tempa, tempb))*(kernelT[b + n] / denomT);
						
					}
					output.at<G>(i, j) = (G)sum1;

				}
			}
			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							
							
							kernelvalue = kernelS[a + n] / denomS  *(kernelT[b + n] / denomT);
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b));
							sum2 += kernelvalue;

						}
					}
				}

				output.at<G>(i, j) = (G)(sum1 / sum2);

			}
		}
	}
	return output;
}