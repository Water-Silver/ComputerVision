#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

using namespace cv;

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Bilateral_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt);
Mat Bilateral_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r,const char *opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

												// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

	// Denoise, using gaussian filter
	Mat Denoised_Gray =Bilateral_Gray(noise_Gray, 3, 10, 10,10, "zero-padding");
	Mat Denoised_RGB = Bilateral_RGB(noise_RGB, 3, 10, 10,10, "adjustkernel");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (Grayscale)", noise_Gray);

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (RGB)", Denoised_RGB);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}
Mat Bilateral_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt) {

	Mat kernel;
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa, tempb;
	double kernelvalue;

	kernel = Mat::ones(kernel_size, kernel_size, CV_64F);
	// Initialiazing Gaussian Kernel Matrix
	// Fill code to initialize Gaussian filter kernel matrix

	Mat output = Mat::zeros(row, col, input.type());
	
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			double value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<double>(a + n, b + n) = value1;
		}
	}


	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) {
				double sum1 = 0.0;
				double Dsum = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							kernelvalue = kernel.at<double>(a + n, b + n) * exp(- (pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2)) / (2 * pow(sigma_r, 2)));
							Dsum += kernel.at<double>(a + n, b + n) * exp(- (pow(input.at<double>(i,j) - input.at<double>(i+a, j+b) ,2)) / (double) (2 * pow(sigma_r, 2))) ;
							sum1 += kernelvalue * (input.at<double>(i + a, j + b));
							
						}
					}
				}
			
				output.at<double>(i, j) = (double)sum1/(Dsum);

			}

			else if (!strcmp(opt, "mirroring")) {
				double sum1 = 0.0;
				double Dsum = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
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
						
						kernelvalue = kernel.at<double>(a + n, b + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(tempa, tempb), 2) / (2 * pow(sigma_r, 2)));
						Dsum += kernel.at<double>(a + n, b + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(tempa,tempb), 2) / (2 * pow(sigma_r, 2)));
						sum1 += kernelvalue * (input.at<double>(tempa, tempb));
					
					}
					output.at<double>(i, j) = (double)sum1/Dsum;

				}

			}
			else if (!strcmp(opt, "adjustkernel")) {
				double Dsum = 0.0;
				double sum1 = 0.0;
				
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							kernelvalue = kernel.at<double>(a + n, b + n) * exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2)) / (2 * pow(sigma_r, 2)));
							Dsum += kernel.at<double>(a + n, b + n) * exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2)) / (double)(2 * pow(sigma_r, 2)));
							sum1 += kernelvalue * (input.at<double>(i + a, j + b));
						}
					}
				}

				output.at<double>(i, j) = (double)sum1 / Dsum;

			}
		}
	}
	return output;
}

Mat Bilateral_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa, tempb;
	
	double kernelvalue;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_64FC3);

	Mat output = Mat::zeros(row, col, CV_64FC3);

	
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			double value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<double>(a + n, b + n) = value1;
			
		}
	}

	

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) {
				double sum1_r = 0.0;
				double sum1_g = 0.0;
				double sum1_b = 0.0;
				double Dsum = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							kernelvalue = kernel.at<double>(a + n, b + n) * exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2)) / (2 * pow(sigma_r, 2)));
							Dsum += kernel.at<double>(a + n, b + n) * exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2)) / (double)(2 * pow(sigma_r, 2)));
							sum1_r += kernelvalue *(double) (input.at<Vec3d>(i + a, j + b)[0]);
							sum1_g += kernelvalue*(double)(input.at<Vec3d>(i + a, j + b)[1]);
							sum1_b += kernelvalue*(double)(input.at<Vec3d>(i + a, j + b)[2]);
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)sum1_r/Dsum;
				output.at<Vec3d>(i, j)[1] = (double)sum1_g/Dsum;
				output.at<Vec3d>(i, j)[2] = (double)sum1_b/Dsum;

			}

			else if (!strcmp(opt, "mirroring")) {
				double sum1_r = 0.0;
				double sum1_g = 0.0;
				double sum1_b = 0.0;
				double Dsum = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {


						if (i + a > row - 1) {  //mirroring for the border pixels
							tempa = i - a;
						}
						else if (i + a < 0) {
							tempa = -(i + a);
						}
						else {
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
						kernelvalue = kernel.at<double>(a + n, b + n) * exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2)) / (2 * pow(sigma_r, 2)));
						Dsum += kernel.at<double>(a + n, b + n) * exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2)) / (double)(2 * pow(sigma_r, 2)));
						
						sum1_r += kernelvalue*(double)(input.at<Vec3d>(tempa, tempb)[0]);
						sum1_g += kernelvalue*(double)(input.at<Vec3d>(tempa, tempb)[1]);
						sum1_b += kernelvalue*(double)(input.at<Vec3d>(tempa, tempb)[2]);
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)sum1_r/Dsum;
				output.at<Vec3d>(i, j)[1] = (double)sum1_g/Dsum;
				output.at<Vec3d>(i, j)[2] = (double)sum1_b/Dsum;

			}


			else if (!strcmp(opt, "adjustkernel")) {
				double sum1_r = 0.0;
				double sum1_g = 0.0;
				double sum1_b = 0.0;
				double Dsum = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							
							kernelvalue = kernel.at<double>(a + n, b + n) * exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2)) / (2 * pow(sigma_r, 2)));
							Dsum += kernel.at<double>(a + n, b + n) * exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2)) / (double)(2 * pow(sigma_r, 2)));
						
							sum1_r += kernelvalue*(double)(input.at<Vec3d>(i + a, j + b)[0]);
							sum1_g += kernelvalue*(double)(input.at<Vec3d>(i + a, j + b)[1]);
							sum1_b += kernelvalue*(double)(input.at<Vec3d>(i + a, j + b)[2]);
							
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)sum1_r / Dsum;
				output.at<Vec3d>(i, j)[1] = (double)sum1_g / Dsum;
				output.at<Vec3d>(i, j)[2] = (double)sum1_b / Dsum;

			}
		}
	}
	return output;
}