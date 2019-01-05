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

Mat unsharp_gray(const Mat input, int n, float sigmaT, float sigmaS, const char* opt,float k);

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

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	output = unsharp_gray(input_gray, 1, 1, 1,"adjustkernel", 0.5);

	namedWindow("Unsharp Gray",WINDOW_AUTOSIZE);
	imshow("Unsharp_Gray", output);


	waitKey(0);

	return 0;
}


Mat unsharp_gray(const Mat input, int n, float sigmaT, float sigmaS, const char* opt,float k) {

	Mat kernel; // gausiaan L
	
	float Idata[] = { 0,0,0,0,1,0,0,0,0 }; 
	Mat I(3, 3, CV_32F, Idata);
	
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Kernel Matrix 
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F);

	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}
	

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (!strcmp(opt, "zero-paddle")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
	
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							kernelvalue = kernel.at<float>(a + n, b + n); 
							sum1 += (I.at<float>(a + n, b + n) - kernelvalue * k) /(float) (1 - k) *(float)(input.at<G>(i + a, j + b));
						}
							}
						}

				if (sum1 < 0)sum1 = 0;
				if (sum1 > 255) sum1 = 255;
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
						kernelvalue = kernel.at<float>(a + n, b + n); // 가우시안 상수
						sum1 += (I.at<float>(a + n, b + n) - kernelvalue*k) / (float)(1 - k) * (float)(input.at<G>(tempa, tempb));
						

					}
					output.at<G>(i, j) = (G)sum1;

				}
			}
			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				float iw = 0.0;
				
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

			
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							kernelvalue = kernel.at<float>(a + n, b + n);
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b));
							sum2 += kernelvalue;
							iw += I.at<float>(a + n, b + n)*(float)(input.at<G>(i + a, j + b));
							
						}
					}
				}
				float out = iw - (sum1 / sum2)*k;
				if (out < 0) out = 0;
				if (out > 255) out = 255;
					output.at<G>(i, j) = (G)(out)/(1-k);

			}
		}
	}
	return output;
}