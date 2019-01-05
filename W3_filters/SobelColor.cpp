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

Mat sobelfilter_rgb(const Mat input);

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
	output = sobelfilter_rgb(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);


	waitKey(0);

	return 0;
}


Mat sobelfilter_rgb(const Mat input) {

	//Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

			   // Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
			   //Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
	float Sxdata[] = { -1,0,1,-2,0,2,-1,0,1 };
	float Sydata[] = { -1,-2,-1,0,0,0,1,2,1 };
	
	Mat Sx(3, 3, CV_32F, Sxdata);
	Mat Sy(3, 3, CV_32F, Sydata);
	int tempa;
	int tempb;
	//printf("%lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf", Sx.at<float>(0, 0), Sx.at<float>(0, 1), Sx.at<float>(0, 2), Sx.at<float>(1, 0), Sx.at<float>(2, 0), Sx.at<float>(1, 1), Sx.at<float>(1, 2), Sx.at<float>(2, 1), Sx.at<float>(2, 2));

	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
			float sum2 = 0.0;
			float sum1_r_Sx = 0.0;
			float sum1_g_Sx = 0.0;
			float sum1_b_Sx = 0.0;
			float sum1_r_Sy = 0.0;
			float sum1_g_Sy = 0.0;
			float sum1_b_Sy = 0.0;
			float sum3_r = 0.0;
			float sum3_g= 0.0;
			float sum3_b=0.0 ;

			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {

					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 

					if (i + a > row - 1) {  //mirroring for the border pixels
						tempa = i-a ;
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
					sum1_r_Sx += Sx.at<float>(a + 1, b + 1)*(float)(input.at<C>(tempa, tempb)[0]);
					sum1_g_Sx += Sx.at<float>(a + 1, b + 1)*(float)(input.at<C>(tempa, tempb)[1]);
					sum1_b_Sx += Sx.at<float>(a + 1, b + 1)*(float)(input.at<C>(tempa, tempb)[2]);
					sum1_r_Sy += Sy.at<float>(a + 1, b + 1)*(float)(input.at<C>(tempa, tempb)[0]);
					sum1_g_Sy += Sy.at<float>(a + 1, b + 1)*(float)(input.at<C>(tempa, tempb)[1]);
					sum1_b_Sy += Sy.at<float>(a + 1, b + 1)*(float)(input.at<C>(tempa, tempb)[2]);

				}
			}
			sum3_r = pow(sum1_r_Sx, 2) + pow(sum1_r_Sy, 2);
			sum3_g = pow(sum1_g_Sx, 2) + pow(sum1_g_Sy, 2);
			sum3_b = pow(sum1_b_Sx, 2) + pow(sum1_b_Sy, 2);

			sum3_r = sqrt(sum3_r);
			sum3_g = sqrt(sum3_g);
			sum3_b = sqrt(sum3_b);
		
			//printf("%lf\t", sum1);
			output.at<C>(i, j)[0] = (G)(sum3_r);
			output.at<C>(i, j)[1] = (G)(sum3_g);
			output.at<C>(i, j)[2] = (G)(sum3_b);
		}

	}

	return output;


}
