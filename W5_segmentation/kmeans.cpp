#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3


#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#endif
using namespace cv;
Mat KmeansGray(const Mat input, double sigma, int InputNum);
Mat KmeansColor(const Mat input, double sigma,int InputNum);

// Note that this code is for the case when an input data is a color value.
int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat outputGray1;
	Mat outputGray2;
	Mat outputColor1;
	Mat outputColor2;

	float sigma = 5;

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);

	namedWindow("OriginalGray", WINDOW_AUTOSIZE);
	imshow("OriginalGray", input_gray);


	outputGray1 = KmeansGray(input_gray, sigma, 1);
	outputGray2 = KmeansGray(input_gray, sigma, 3);
	outputColor1 = KmeansColor(input, sigma, 3);
	outputColor2 = KmeansColor(input, sigma, 5);

	namedWindow("KmeansGray1", WINDOW_AUTOSIZE);
	imshow("KmeansGray1", outputGray1);
	namedWindow("KmeansGray2", WINDOW_AUTOSIZE);
	imshow("KmeansGray2", outputGray2);

	namedWindow("KmeansColor1", WINDOW_AUTOSIZE);
    imshow("KmeansColor1", outputColor1);
	namedWindow("KmeansColor2", WINDOW_AUTOSIZE);
	imshow("KmeansColor2", outputColor2);


	waitKey(0);

	return 0;
}

Mat KmeansGray(const Mat input, double sigma, int i) {

	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;
	Mat new_image;
	if (i == 1) {
		Mat samples_gray(input.rows * input.cols, 1, CV_32F);
		for (int x = 0; x < input.rows; x++)
			for (int y = 0; y < input.cols; y++)
				samples_gray.at<float>(x + y*input.rows, 0) =(float) input.at<G>(x,y);

		kmeans(samples_gray, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
		new_image = Mat::zeros(input.rows, input.cols, CV_32F);
		for (int x = 0; x < input.rows; x++) {
			for (int y = 0; y < input.cols; y++){
				int cluster_idx = labels.at<int>(x + y*input.rows,0);

				//fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
				new_image.at <float>(x, y) = (float)centers.at<float>(cluster_idx, 0)/255.0;

			}
		}
		return new_image;
	}
	else { // i = 3;
		Mat samples2_gray(input.rows * input.cols, 3, CV_32F); //size 크기 일차 배열 rgb 각각 3채널
		for (int x = 0; x < input.rows; x++) {
			for (int y = 0; y < input.cols; y++) {
				samples2_gray.at<float>(x + y*input.rows, 0) = (float)input.at<G>(x, y)/255.0; //
				//printf("%f", (float)input.at<G>(x, y) );
				samples2_gray.at<float>(x + y*input.rows, 1) = ((float)x / (float)input.cols) / sigma;
				samples2_gray.at<float>(x + y*input.rows, 2) = ((float)y / (float)input.rows) / sigma;
			}
		}
		kmeans(samples2_gray, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
		new_image = Mat::zeros(input.rows, input.cols, CV_32F);
		for (int x = 0; x < input.rows; x++)
			for (int y = 0; y < input.cols; y++)
			{
				int cluster_idx = labels.at<int>(x + y*input.rows, 0);

				new_image.at <float>(x, y) = (float) centers.at<float>(cluster_idx, 0);
				
				
			}

		return new_image;

	}


}


Mat KmeansColor(const Mat input, double sigma, int i) {


	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;

	if (i == 3) {
		Mat samples(input.rows * input.cols, 3, CV_32F); //size 크기 일차 배열 rgb 각각 3채널
		for (int x = 0; x < input.rows; x++)
			for (int y = 0; y < input.cols; y++)
				for (int z = 0; z < 3; z++)
					samples.at<float>(x + y*input.rows, z) = input.at<Vec3b>(x, y)[z]; // 

		kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
		Mat new_image(input.size(), input.type());
		for (int x = 0; x < input.rows; x++)
			for (int y = 0; y < input.cols; y++)
			{
				int cluster_idx = labels.at<int>(x + y*input.rows, 0);
				//printf("%d", cluster_idx);
				//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
				new_image.at <Vec3b>(x, y)[0] = centers.at<float>(cluster_idx, 0);
				new_image.at <Vec3b>(x, y)[1] = centers.at<float>(cluster_idx, 1);
				new_image.at <Vec3b>(x, y)[2] = centers.at<float>(cluster_idx, 2);
				//printf("%f\n", centers.at<float>(cluster_idx, 2));
				//printf("%d", centers.at<float>(cluster_idx, 0));
			}

		return new_image;
	}
	else {
		Mat samples2(input.rows * input.cols, 5, CV_32F); //size 크기 일차 배열 rgb 각각 3채널
		for (int x = 0; x < input.rows; x++) {
			for (int y = 0; y < input.cols; y++) {
				for (int z = 0; z < 3; z++) {
					samples2.at<float>(x + y*input.rows, z) = (float)input.at<Vec3b>(x, y)[z] / 255.0; //
				}
				samples2.at<float>(x + y*input.rows, 3) = ((float)x / (float)input.cols) / sigma;
				samples2.at<float>(x + y*input.rows, 4) = ((float)y / (float)input.rows) / sigma;
			}
		}
		kmeans(samples2, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
		Mat new_image(input.size(), input.type());
		for (int x = 0; x < input.rows; x++)
			for (int y = 0; y < input.cols; y++)
			{
				int cluster_idx = labels.at<int>(x + y*input.rows, 0);

				//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
				new_image.at <Vec3b>(x, y)[0] = centers.at<float>(cluster_idx, 0)*255;
				new_image.at <Vec3b>(x, y)[1] = centers.at<float>(cluster_idx, 1)*255;
				new_image.at <Vec3b>(x, y)[2] = centers.at<float>(cluster_idx, 2)*255;
				//printf("%f\n", centers.at<float>(cluster_idx, 2));
				//printf("%d", centers.at<float>(cluster_idx, 0));
			}

		return new_image;
	
	}



	
	//imshow("clustered image", new_image);
}
