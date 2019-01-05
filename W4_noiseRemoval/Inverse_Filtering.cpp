
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
Mat Inversefilter(const Mat input, int n, double sigma_t, double sigma_s, double d);
Mat FourierTransform(const Mat input, int m, int n, bool inverse);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	//Gaussian smoothing parameters
	int window_radius = 7;
	double sigma_t = 5.0;
	double sigma_s = 5.0;

	//AWGN noise variance
	double noise_var = 0.03;

	//Deconvolution threshold
	double decon_thres = 0.1;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point
	printf("%lf\n", input_gray.at<double>(200, 200));

	Mat h_f = Gaussianfilter(input_gray, window_radius, sigma_t, sigma_s);	// h(x,y) * f(x,y)
	Mat g = Add_Gaussian_noise(h_f, 0, noise_var);		//					+ n(x, y)

	Mat F = Inversefilter(g, window_radius, sigma_t, sigma_s, decon_thres);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Gaussian Noise", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise", g);

	namedWindow("Deconvolution result", WINDOW_AUTOSIZE);
	imshow("Deconvolution result", F);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, CV_64F);
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s) {
	
	int row = input.rows;
	int col = input.cols;
	double kernelvalue;
	double sum1=0;
	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);
	Mat output = Mat::zeros(row, col, CV_64F);

	// convolution with zero padding
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			sum1 = 0;
			for (int x = -n; x <= n; x++) { // for each kernel window
				for (int y = -n; y <= n; y++) {

					/* Gaussian filter with "zero-padding" boundary process:

					Fill the code:
					*/
					if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
						kernelvalue = kernel.at<double>(x + n, y + n);
						
						sum1 += kernelvalue * (double)(input.at<double>(i + x, j + y));
						
					}
				}
			}
			output.at<double>(i, j) = (double)sum1;
			
		}
	}
	

	return output;
}

Mat Inversefilter(const Mat input, int n, double sigma_t, double sigma_s, double d) {

	int row = input.rows;
	int col = input.cols;

	

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);
	Mat G = FourierTransform(input, row, col, 0);
	Mat H = FourierTransform(kernel, row, col, 0);
	Mat F = G.clone();
	// Fill the code to
	// Perform Fourier Transform on Noise Image(G) and Gaussian Kernel(H)

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			double h1 = H.at<Vec2d>(i, j)[0];
			double h2 = H.at<Vec2d>(i, j)[1];
			double g1 = G.at<Vec2d>(i, j)[0];
			double g2 = G.at<Vec2d>(i, j)[1];

			double magH = sqrt(h1*h1 + h2*h2);
			/* Element-wise divide for compute F (F = G / H)
			
			Fill the code:
			*/
			if (magH >= d) {
				F.at<Vec2d>(i, j)[0] = (h1 *g1 + h2 * g2) / (magH *magH);
				F.at<Vec2d>(i, j)[1] = (h1 * g2 - h2*g1) / (magH * magH);
			}
		


		}
	}

	// Fill the code to perform Inverse Fourier Transform
	F = FourierTransform(F, row, col, 1);

	return F;
}

Mat FourierTransform(const Mat input, int m, int n, bool inverse) {

	//expand input image to optimal size
	Mat padded;
	copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat Transformed;



	// Applying DFT
	if (!inverse) {
		dft(padded, Transformed, DFT_COMPLEX_OUTPUT);
	}
	// Reconstructing original image from the DFT coefficients
	else {
		idft(padded, Transformed, DFT_SCALE | DFT_REAL_OUTPUT);
//		normalize(Transformed, Transformed, 0, 1, CV_MINMAX);
	}

	return Transformed;
}

Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize) {
	
	Mat kernel;
	int kernel_size = (2 * n + 1);
	double denom = 0.0;
	kernel = Mat::ones(kernel_size, kernel_size, CV_64F);
	// Initialiazing Gaussian Kernel Matrix
	// Fill code to initialize Gaussian filter kernel matrix

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			double value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<double>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	

	// if "normalize" is true
	// return normalized Guassian Kernel
	// else, return unnormalized one
	if (normalize) {
		// Fill code to normalize kernel
		for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
			for (int b = -n; b <= n; b++) {
				kernel.at<double>(a + n, b + n) /= denom;
			}
		}

	}

	return kernel;
}