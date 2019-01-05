/* team project credits to Ina Shin*/
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <string>
#define IM_TYPE	CV_8UC3


#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#endif
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

using namespace cv;
using namespace std;

Mat meanfilter(const Mat input, int n);
Mat KmeansGray(const Mat input, double sigma);
vector<Point2f> MatToVec(const Mat input);
void NonMaximum_Suppression(const Mat input, Mat corner_mat, int radius);
Mat Mirroring(const Mat input, int n);
Mat cal_perpec(Point2f * scr, Point2f * des);
Mat warp_perpec(Mat src, Mat T, Size size);
Mat adaptive_thres(const Mat input, int n, float bnumber);
Mat unsharp_gray(const Mat input, int n, float sigmaT, float sigmaS,float k);


int main() {

	Mat input = imread("trial3.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray, input_visual;
	Mat output, output_norm, corner_mat;
	Mat final_output;
	vector<Point2f> points;
	float sigma = 5;


	if (!input.data) {
		printf_s("Could not open\n");
		return -1;
	}

	int forresize_x = input.cols / 770;
	resize(input, input, Size(input.cols / forresize_x, input.rows / forresize_x));


	int row = input.rows;
	int col = input.cols;

	int blockSize = 12;
	int apertureSize = 5;
	double k = 0.0001;
	int corner_num = 0;

	corner_mat = Mat::zeros(row, col, CV_8U);

	bool NonMaxSupp = true;
	bool Subpixel = true;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale	
	
	Mat outputGray2 = KmeansGray(input_gray, sigma);
	namedWindow("1.kmeans", WINDOW_AUTOSIZE);
	imshow("1.kmeans", outputGray2);
	imwrite("1.keans.png", outputGray2);


	for (int i = 0; i <8; i++) {
		outputGray2 = meanfilter(outputGray2, 10); 

	}

	namedWindow("2.meanfiltered", WINDOW_AUTOSIZE);
	imshow("2.meanfiltered", outputGray2);
	imwrite("2.meanfiltered.png", outputGray2);

	cornerHarris(outputGray2, output, blockSize, apertureSize, k, BORDER_DEFAULT);

	//Scale the Harris response map 'output' from 0 to 1.
	//This is for display purpose only.
	normalize(output, output_norm, 0, 1.0, NORM_MINMAX);
	imwrite("3.HarrisResponse.png", output_norm);
	namedWindow("3.Harris Response", WINDOW_AUTOSIZE);
	imshow("3.Harris Response", output_norm);


	//Threshold the Harris corner response.
	//corner_mat = 1 for corner, 0 otherwise.
	input_visual = input.clone();
	double minVal, maxVal;		Point minLoc, maxLoc;
	minMaxLoc(output, &minVal, &maxVal, &minLoc, &maxLoc);


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (output.at<float>(i, j) > 0.01*maxVal)
			{
				
				circle(input_visual, Point(j, i), 2, Scalar(0, 0, 255), 1, 8, 0);

				corner_mat.at<uchar>(i, j) = 1;
				corner_num++;
			}

			else
				output.at<float>(i, j) = 0.0;
		}
	}
	//printf("After cornerHarris, corner number = %d\n\n", corner_num);
	imwrite("4.HarrisCorner.png", input_visual);
	//namedWindow("Harris Corner", WINDOW_AUTOSIZE);
	//imshow("Harris Corner", input_visual);



	//Non-maximum suppression
	if (NonMaxSupp)
	{
		NonMaximum_Suppression(output, corner_mat, 2);

		corner_num = 0;
		input_visual = input.clone();
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (corner_mat.at<uchar>(i, j) == 1) {
					//input_visual.at<Vec3b>(i, j)[0] = 0;		input_visual.at<Vec3b>(i, j)[1] = 0;	input_visual.at<Vec3b>(i, j)[2] = 255;					
					circle(input_visual, Point(j, i), 2, Scalar(0, 0, 255), 1, 8, 0);	//You can also use this function of drawing a circle. For details, search 'circle' in OpenCV.
					corner_num++;
				}
			}
		}

		//printf("After non-maximum suppression, corner number = %d\n\n", corner_num);
		//namedWindow("Harris Corner (Non-max)", WINDOW_AUTOSIZE);
		//imshow("Harris Corner (Non-max)", input_visual);
		imwrite("5.HarrisCorner_Non-max.png", input_visual);
	}

	//Sub-pixel refinement for detected corners
	if (Subpixel)
	{
		Size subPixWinSize(3, 3);
		TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);

		points = MatToVec(corner_mat);

		cornerSubPix(input_gray, points, subPixWinSize, Size(-1, -1), termcrit);

		//Display the set of corners
		input_visual = input.clone();
		for (int k = 0; k < points.size(); k++) {

			int x = points[k].x;
			int y = points[k].y;

			if (x<0 || x>col - 1 || y<0 || y>row - 1)
			{
				points.pop_back();
				continue;
			}

			circle(input_visual, Point(x, y), 2, Scalar(0, 0, 255), 1, 8, 0);	//You can also use this function of drawing a circle. For details, search 'circle' in OpenCV.
		}

		//printf("After subpixel-refinement, corner number = %d\n\n", points.size());
		//namedWindow("Harris Corner (subpixel)", WINDOW_AUTOSIZE);
		//imshow("Harris Corner (subpixel)", input_visual);
		imwrite("6.HarrisCorner_subpixel.png", input_visual);
	}



	//////////////////////****** ******////////////////////////////
	
	//유의미한 차이를 확인하기 위함
	int for_compare = input.cols / 100;

	Point2f points_sorted[4]; //0~3 순서대로 위 왼쪽 , 위 오른쪽, 아래 왼쪽, 아래 오른쪽

	//위 아래 비교
	int kk =1;
	double comp = points[0].y; // 기준
	double check;


	//유의미한 차이가 있을 때 즉 위 아래
	for (;;) {
		check = comp - points[kk].y;
		if (abs(check) >  for_compare)
			break;
		kk++;
	}

	//comp가 더 큰 경우 - 0이 아래
	if (check > 0) {
		points_sorted[2] = points[0];
		points_sorted[0] = points[kk];
	}
	//comp가 더 작은 경우 - 0이 위
	else {
		points_sorted[0] = points[0];
		points_sorted[2] = points[kk];
	}


	int j = 1;
	while(j<4) {
		if (j == kk) {
		}
		else
		{
			//역시 0이랑 비교, 유의미한 차이가 을 시, 위 아래 반대
			check = comp - points[j].y;
			if (abs(check) >  for_compare) {
				if (check > 0)
					points_sorted[1] = points[j];
				else
					points_sorted[3] = points[j];
			}
			//유의미한 차이가 없을 시
			else {
				//0과 반대위치에 있는 kk와 비교
				check = points[kk].y - points[j].y;
				if (check > 0)
					points_sorted[1] = points[j];
				else
					points_sorted[3] = points[j];
			}

		}
		j++;

	}



	//오른 왼 비교
	Point2f temp;
	if ((points_sorted[0].x - points_sorted[1].x) > 0) {
		temp = points_sorted[0];
		points_sorted[0] = points_sorted[1];
		points_sorted[1] = temp;
	}

	if ((points_sorted[2].x - points_sorted[3].x) > 0) {
		temp = points_sorted[2];
		points_sorted[2] = points_sorted[3];
		points_sorted[3] = temp;
	}


	// 영역을 와핑하여 저장하고자 하는 영역의 네개의 점
	double w1 = sqrt(pow(points_sorted[3].x - points_sorted[2].x, 2)
		+ pow(points_sorted[3].x - points_sorted[2].x, 2));
	double w2 = sqrt(pow(points_sorted[1].x - points_sorted[0].x, 2)
		+ pow(points_sorted[1].x - points_sorted[0].x, 2));

	double h1 = sqrt(pow(points_sorted[1].y - points_sorted[3].y, 2)
		+ pow(points_sorted[1].y - points_sorted[3].y, 2));
	double h2 = sqrt(pow(points_sorted[0].y - points_sorted[2].y, 2)
		+ pow(points_sorted[0].y - points_sorted[2].y, 2));



	double maxWidth = (w1 < w2) ? w1 : w2;
	double maxHeight = (h1 < h2) ? h1 : h2;

	Point2f scr[4], dst[4];
	//top left
	scr[0] = Point2f(points_sorted[0].x, points_sorted[0].y);
	//top right
	scr[1] = Point2f(points_sorted[1].x, points_sorted[1].y);
	//bottom right
	scr[2] = Point2f(points_sorted[3].x, points_sorted[3].y);
	//bottom left
	scr[3] = Point2f(points_sorted[2].x, points_sorted[2].y);

	dst[0] = Point2f(0, 0);
	dst[1] = Point2f(maxWidth - 1, 0);
	dst[2] = Point2f(maxWidth - 1, maxHeight - 1);
	dst[3] = Point2f(0, maxHeight - 1);

	Mat transformMatrix;
	

	transformMatrix = cal_perpec(scr, dst);

	Size size = Size(maxWidth, maxHeight);
	final_output = warp_perpec(input, transformMatrix, size);




	////////////////////////***********warp 후의 이미지 adjust/////////////////////////////////////////
	Mat final_output_adj;
	int adjust=0;

	cvtColor(final_output, final_output_adj, CV_RGB2GRAY); // Converting image to gray

	printf_s("How do you want to save the image?\n");
	while(adjust!= 1 && adjust != 2){
		printf_s("Type in 1 for binary, 2 for grayscale: ");
		scanf_s("%d", &adjust);
		if (adjust == 1) {
			printf_s("\nYou choosed binary image.\n");
			final_output_adj = adaptive_thres(final_output_adj, 4, 0.95); //Fix with uniform mean filtering with zero paddle
			imwrite("7.Final_Image_binary.png", final_output_adj);
		}
		else if (adjust == 2) {
			printf_s("\nYou choosed grayscale image.\n");
			final_output_adj = unsharp_gray(final_output_adj, 1, 1, 1, 0.5);
			imwrite("7.Final_Image_grayscale.png", final_output_adj);
		}
		else
			printf_s("You typed in an unavailable number.\n\n");

	}


	
	namedWindow("warperspective", WINDOW_AUTOSIZE);
	imshow("warperspective", final_output_adj);

	waitKey(0);

	return 0;
}


Mat cal_perpec(Point2f * scr, Point2f * des) {

	Mat input(8, 1, CV_32F);
	Mat forinv(8, 8, CV_32F, Scalar(0));
	Mat ans(3, 3, CV_32F);
	Mat temp, perpec;

	for (int i = 0; i < 4; i++) {

		input.at<float>(2 * i, 0) = des[i].x;
		input.at<float>(2 * i + 1, 0) = des[i].y;

		forinv.at<float>(2 * i, 0) = scr[i].x;
		forinv.at<float>(2 * i, 1) = scr[i].y;
		forinv.at<float>(2 * i, 2) = 1;
		forinv.at<float>(2 * i, 3) = 0;
		forinv.at<float>(2 * i, 4) = 0;
		forinv.at<float>(2 * i, 5) = 0;
		forinv.at<float>(2 * i, 6) = -scr[i].x * des[i].x;
		forinv.at<float>(2 * i, 7) = -des[i].x * scr[i].y;

		forinv.at<float>(2 * i + 1, 0) = 0;
		forinv.at<float>(2 * i + 1, 1) = 0;
		forinv.at<float>(2 * i + 1, 2) = 0;
		forinv.at<float>(2 * i + 1, 3) = scr[i].x;
		forinv.at<float>(2 * i + 1, 4) = scr[i].y;
		forinv.at<float>(2 * i + 1, 5) = 1;
		forinv.at<float>(2 * i + 1, 6) = -scr[i].x * des[i].y;
		forinv.at<float>(2 * i + 1, 7) = -scr[i].y * des[i].y;

	}


	temp = forinv.inv(DECOMP_SVD);
	perpec = temp * input; //8*1 상태


	ans.at<float>(0, 0) = perpec.at<float>(0, 0);
	ans.at<float>(0, 1) = perpec.at<float>(1, 0);
	ans.at<float>(0, 2) = perpec.at<float>(2, 0);
	ans.at<float>(1, 0) = perpec.at<float>(3, 0);
	ans.at<float>(1, 1) = perpec.at<float>(4, 0);
	ans.at<float>(1, 2) = perpec.at<float>(5, 0);
	ans.at<float>(2, 0) = perpec.at<float>(6, 0);
	ans.at<float>(2, 1) = perpec.at<float>(7, 0);
	ans.at<float>(2, 2) = 1;

	return ans;
}


Mat warp_perpec(Mat src, Mat T, Size size) {
	
	int x=0, y=0;
	Mat dst = src.clone();
	resize(dst, dst, size);


	T = T.inv();

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {

			y = (int)((T.at<float>(0, 0) *j + T.at<float>(0, 1)*i + T.at<float>(0, 2))
				/ (T.at<float>(2, 0) *j + T.at<float>(2, 1)*i+ T.at<float>(2, 2)));


			x = (int)((T.at<float>(1, 0) *j + T.at<float>(1, 1)*i + T.at<float>(1, 2))
			 /(T.at<float>(2, 0) *j+ T.at<float>(2, 1)*i + T.at<float>(2, 2)));

			for (int k = 0; k < 3; k++) {
				if(x<src.rows && y<src.cols)
					dst.at<C>(i, j)[k] = src.at<C>(x, y)[k];

			}
			

			
		}
	}

	return dst;

}



Mat adaptive_thres(const Mat input, int n, float bnumber) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);

	// Initialiazing Kernel Matrix 
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	float kernelvalue = kernel.at<float>(0, 0); 
	// To simplify, as the filter is uniform. All elements of the kernel value are same.

	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) { //for each pixel in the output
		for (int j = 0; j < col; j++) {

			float sum1 = 0.0;
			// Fill code that finds the mean intensity using uniform mean filtering with zero paddle border process.
			for (int a = -n; a <= n; a++) { // for each kernel window
				for (int b = -n; b <= n; b++) {

					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
					}
				}
			}

			float temp = bnumber * (G)sum1;


			if (input.at<G>(i, j) > temp)
				output.at<G>(i, j) = 255;
			else
				output.at<G>(i, j) = 0;


		}
	}
	return output;
}



vector<Point2f> MatToVec(const Mat input)
{
	vector<Point2f> points;

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<uchar>(i, j) == 1) {
				points.push_back(Point2f((float)j, (float)i));
			}
		}
	}

	return points;
}

//corner_mat = 1 for corner, 0 otherwise.
void NonMaximum_Suppression(const Mat input, Mat corner_mat, int radius)
{
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * radius + 1);
	float max = 0.0;

	Mat input_mirror = Mirroring(input, radius);

	//std::cout << "Could not open" << std::endl;


	//std::cout << input << std::endl; 0 이 많지만 다 0은 아님

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			max = 0;
			for (int a = -radius; a <= radius; a++) { // for each kernel window
				for (int b = -radius; b <= radius; b++) {
					//printf("%d %d %d %d %lf\n", i, j, i + a + radius, j + b + radius, input_mirror.at<float>(i + a + radius, j + b + radius));
					if ((float)input_mirror.at<float>(i + a + radius, j + b + radius) - max > 0) {
						max = (float)input_mirror.at<float>(i + a + radius, j + b + radius);

						//printf("found max %d %d %d %d %lf\n",i,j, i + a+radius, j+b + radius,max);

					}
				}
			}

			// printf("%d %d %lf %lf\n",i,j, (float)input_mirror.at<float>(i + radius, j + radius),max);
			//printf("%lf\n", input_mirror.at<float>(i+radius,j+radius));

			if (max != 0 && input_mirror.at<float>(i + radius, j + radius) >= max) {
				corner_mat.at<uchar>(i, j) = (uchar) 1.0;
			}
			else {
				corner_mat.at<uchar>(i, j) = (uchar) 0.0;
			}

		}

	}




	//return input;
}

Mat Mirroring(const Mat input, int n)
{
	int row = input.rows;
	int col = input.cols;

	Mat input2 = Mat::zeros(row + 2 * n, col + 2 * n, input.type());
	int row2 = input2.rows;
	int col2 = input2.cols;

	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			input2.at<float>(i, j) = input.at<float>(i - n, j - n);
		}
	}
	for (int i = n; i < row + n; i++) {
		for (int j = 0; j < n; j++) {
			input2.at<float>(i, j) = input2.at<float>(i, 2 * n - j);
		}
		for (int j = col + n; j < col2; j++) {
			//input2.at<float>(i, j) = input2.at<float>(i, 2 * col - 2 + 2 * n - j);
			input2.at<float>(i, j) = input2.at<float>(i, 2 * col + 2 * n - j);


		}
	}
	for (int j = 0; j < col2; j++) {
		for (int i = 0; i < n; i++) {
			input2.at<float>(i, j) = input2.at<float>(2 * n - i, j);
		}
		for (int i = row + n; i < row2; i++) {
			//input2.at<float>(i, j) = input2.at<float>(2 * row - 2 + 2 * n - i, j);
			input2.at<float>(i, j) = input2.at<float>(2 * row + 2 * n - i, j);

		}
	}

	return input2;
}


Mat KmeansGray(const Mat input, double sigma) {

	int clusterCount = 2;
	Mat labels;
	int attempts = 5;
	Mat centers;
	Mat new_image;

		Mat samples_gray(input.rows * input.cols, 1, CV_32F);
		for (int x = 0; x < input.rows; x++)
			for (int y = 0; y < input.cols; y++)
				samples_gray.at<float>(x + y * input.rows, 0) = (float)input.at<G>(x, y);

		kmeans(samples_gray, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
		new_image = Mat::zeros(input.rows, input.cols, CV_32F);
		for (int x = 0; x < input.rows; x++) {
			for (int y = 0; y < input.cols; y++) {
				int cluster_idx = labels.at<int>(x + y * input.rows, 0);

				//fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
				new_image.at <float>(x, y) = (float)centers.at<float>(cluster_idx, 0) / 255.0;

			}
		}
		return new_image;

}

Mat meanfilter(const Mat input, int n) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	// Initialiazing Kernel Matrix 
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	float kernelvalue = kernel.at<float>(0, 0);  // To simplify, as the filter is uniform. All elements of the kernel value are same.
												 //printf("%lf", kernelvalue);

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) { //for each pixel in the output
		for (int j = 0; j < col; j++) {
			//boudary processing = adjust kernel 이용 
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += kernelvalue * (float)(input.at<float>(i + a, j + b));
							sum2 += kernelvalue;

						}

					}
				}

				output.at<float>(i, j) = (float)(sum1 / sum2);

		}
	}
	return output;
}


Mat unsharp_gray(const Mat input, int n, float sigmaT, float sigmaS, float k) {

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


	Mat output = Mat::ones(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			//boudary processing = adjust kernel 이용 
				float sum1 = 0.0;
				float sum2 = 0.0;
				float iw = 0.0;

				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {


						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							kernelvalue = kernel.at<float>(a + n, b + n);
							sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
							sum2 += kernelvalue;
							iw += I.at<float>(a + n, b + n)*(float)(input.at<G>(i + a, j + b));

						}
					}
				}
				float out = iw - (sum1 / sum2)*k;
				output.at<G>(i, j) = (G)(out) / (1 - k);

		}
	}

	return output;
}
