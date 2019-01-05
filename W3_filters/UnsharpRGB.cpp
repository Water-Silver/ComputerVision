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

Mat unsharp_color(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k);

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
	output = unsharp_color(input, 1, 1, 1, "adjustkernel", 0.5);

	namedWindow("Unsharp Color", WINDOW_AUTOSIZE);
	imshow("Unsharp Color", output);


	waitKey(0);

	return 0;
}


Mat unsharp_color(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k) {

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
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							kernelvalue = kernel.at<float>(a + n, b + n); 
							sum1_r += ((I.at<float>(a + n, b + n) - kernelvalue * k) / (float)(1 - k)) *(float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += ((I.at<float>(a + n, b + n) - kernelvalue * k) / (float)(1 - k) )*(float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += ((I.at<float>(a + n, b + n) - kernelvalue * k) / (float)(1 - k)) *(float)(input.at<C>(i + a, j + b)[2]);
							
						}
					}
				}
				if (sum1_r > 255) sum1_r = 255;
				if (sum1_g > 255) sum1_g = 255;
				if (sum1_b > 255) sum1_b = 255;
				if (sum1_r <0 ) sum1_r = 1;
				if (sum1_g <0 ) sum1_g = 1;
				if (sum1_b <0) sum1_b = 1;
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
						kernelvalue = kernel.at<float>(a + n, b + n); // 가우시안 상수
						sum1_r += (I.at<float>(a + n, b + n) - kernelvalue*k) / (float)(1 - k) * (float)(input.at<C>(tempa, tempb)[0]);
						sum1_g += (I.at<float>(a + n, b + n) - kernelvalue*k) / (float)(1 - k) * (float)(input.at<C>(tempa, tempb)[1]);
						sum1_b += (I.at<float>(a + n, b + n) - kernelvalue*k) / (float)(1 - k) * (float)(input.at<C>(tempa, tempb)[2]);
					
					}
					if (sum1_r > 255) sum1_r = 255;
					if (sum1_g > 255) sum1_g = 255;
					if (sum1_b > 255) sum1_b = 255;
					if (sum1_r <0) sum1_r = 1;
					if (sum1_g <0) sum1_g = 1;
					if (sum1_b <0) sum1_b = 1;
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
				
				float iw_r = 0.0;
				float iw_g = 0.0;
				float iw_b = 0.0;

				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {


						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							kernelvalue = kernel.at<float>(a + n, b + n);
							sum1_r += kernelvalue*(float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += kernelvalue*(float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += kernelvalue*(float)(input.at<C>(i + a, j + b)[2]);
							sum2 += kernelvalue;
							iw_r += I.at<float>(a + n, b + n)*(float)(input.at<C>(i + a, j + b)[0]);
							iw_g += I.at<float>(a + n, b + n)*(float)(input.at<C>(i + a, j + b)[1]);
							iw_b += I.at<float>(a + n, b + n)*(float)(input.at<C>(i + a, j + b)[2]);

						}
					}
				}

				if (sum1_r < 0) sum1_r = 1;
				
				if (sum1_g < 0) sum1_g = 1;
				
				if (sum1_b < 0) sum1_b = 1;
				
				float out_r = iw_r - (sum1_r / sum2)*k;
				float out_g = iw_g - (sum1_g / sum2)*k;
				float out_b = iw_b - (sum1_b / sum2)*k;
				
				if (out_r / (1 - k) < 0) out_r = 1;
				if (out_r / (1 - k)> 255) out_r = 255;
				if (out_g / (1 - k)< 0) out_g = 1;
				if (out_g / (1 - k)> 255)out_g = 255;
				if (out_b / (1 - k)< 0) sum1_b = 1;
				if (out_b / (1 - k)> 255) out_b = 255;
				
				output.at<C>(i, j)[0] = (G)(out_r/(1-k));
				output.at<C>(i, j)[1] = (G)(out_g / (1-k));
				output.at<C>(i, j)[2] = (G)(out_b / (1-k));

			}
		}
	}
	return output;
}