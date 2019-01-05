#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated;

	// Read each image
	input = imread("lena.jpg");

	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// original image
	namedWindow("image");
	imshow("image", input);

	rotated = myrotate<Vec3b>(input, 45, "bilinear");

	// rotated image
	namedWindow("rotated");
	imshow("rotated", rotated);

	waitKey(0);

	return 0;
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180;

	//size of output image
	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	Mat output = Mat::zeros(sq_row, sq_col, input.type()); //initialize output to black screen

	//rotated된 행렬, center를 중심으로 회전
	for (int i = 0; i < sq_row; i++) { 
		for (int j = 0; j < sq_col; j++) { 
			float x = (j - sq_col / 2) * cos(radian) + (i - sq_row / 2) * sin(radian) + col / 2;
			float y = - (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
				if (!strcmp(opt, "nearest")) {
					x = floor(x + 0.5);//floor는 버림 함수이다.
					y = floor(y + 0.5);

					//내림하여 대입한 점의 intensity 출력
					output.at<cv::Vec3b>(j, i)[0] = input.at<cv::Vec3b>(x, y)[0];//blue channel
					output.at<cv::Vec3b>(j, i)[1]= input.at<cv::Vec3b>(x, y)[1];//green channel
					output.at<cv::Vec3b>(j, i)[2]= input.at<cv::Vec3b>(x, y)[2];//red channel

				}
				else if (!strcmp(opt, "bilinear")) {
					//cvResize(input, output, CV_INTER_LINEAR);
					float u = x - floor(x), w = y-floor(y);
					float px, py, qx, qy, rx, ry, sx, sy;
					px = floor(x); py = ceil(y);
					qx = ceil(x); qy = ceil(y);
					rx = floor(x); ry = floor(y);
					sx = ceil(x); sy = floor(y);
					
					float xxxIntensity;
					for (int k = 0;k < 3;k++) {
					float xIntensity = (input.at<cv::Vec3b>(rx, ry)[k])*(1 - u) + (input.at<cv::Vec3b>(sx, sy)[k]) *(u);
					float xxIntensity = ((input.at<cv::Vec3b>(px, py)[k]) *(1 - u) + (input.at<cv::Vec3b>(px, py)[k])*u);
					xxxIntensity = xIntensity * (1 - w) + xxIntensity * w;
					output.at<cv::Vec3b>(j, i)[k] = xxxIntensity;
					}
					

					
				}
			}
		}
	}

	return output;
}