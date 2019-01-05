/* nearestNeighbor(), findPairs(), Mat cal_A, and use of OpenCV functions are newly implemented. Rest were given*/
#include "stdio.h"
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors, int nn);
int second_nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors, int nn);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);
void drawlines(Mat& img, vector<Point3f>& lines, vector<Point2f>& img_pt);
void type2str(int type);
template <typename T>
Mat cal_A(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points, int k);

int main() {

	Mat input1 = imread("img1.png", CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread("img2.png", CV_LOAD_IMAGE_COLOR);
	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures
		4,		// nOctaveLayers
		0.04,	// contrastThreshold
		10,		// edgeThreshold
		1.6		// sigma
	);

	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	//FeatureDetector* detector = new SurfFeatureDetector(
	//	10,	// hessianThreshold
	//	4,		// nOctaves
	//	2,		// nOctaveLayers
	//	true,	// extended
	//	false	// upright
	//);

	//DescriptorExtractor* extractor = new SurfDescriptorExtractor();

	// Create a image for displaying mathing keypoints
	Size sz = Size(input1.size().width + input2.size().width, max(input2.size().height, input1.size().height));
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);	

	input1.copyTo(matchingImage(Rect(0, 0, input1.size().width, input1.size().height)));
	input2.copyTo(matchingImage(Rect(input1.size().width, 0, input2.size().width, input2.size().height)));
	

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1;
	Mat descriptors1;
	detector->detect(input1_gray, keypoints1);
	extractor->compute(input1_gray, keypoints1, descriptors1);
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

	vector<KeyPoint> keypoints2;
	Mat descriptors2;	
	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);
	printf("input2 : %zd keypoints are found.\n", keypoints2.size());

	for (int i = 0; i < keypoints1.size(); i++) {
		KeyPoint kp = keypoints1[i];		
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	for (int i = 0; i < keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i];
		kp.pt.x += input1.size().width;
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints, dstPoints;
	bool crossCheck = true;
	bool ratio_threshold = true;

	//keypoints1 -> srcPoints
	//keypoints2 -> dstPoints
	printf("h");
	findPairs(keypoints1, descriptors1, keypoints2, descriptors2, srcPoints, dstPoints, crossCheck, ratio_threshold);
	printf("%zd keypoints are matched.\n", srcPoints.size());

	// Draw line between nearest neighbor pairs
	for (int i = 0; i < (int)srcPoints.size(); ++i) {
		Point2f pt1 = srcPoints[i];
		Point2f pt2 = dstPoints[i];
		Point from = (Point)pt1;
		Point to = (Point)pt2 + Point(input1.size().width, 0);
		line(matchingImage, from, to, Scalar(0, 0, 255));
	}

	// Display matched image
	namedWindow("Matching");
	imshow("Matching", matchingImage);
	
	
	Mat f_mat = findFundamentalMat(srcPoints, dstPoints, FM_RANSAC, 3, 0.99);

	vector<Point3f> lines1, lines2;
	
	computeCorrespondEpilines(srcPoints, 1, f_mat, lines1);
	computeCorrespondEpilines(dstPoints, 2, f_mat, lines2);

	drawlines(input1, lines1, srcPoints);
	drawlines(input2, lines2, dstPoints);

	namedWindow("Epipolar line (first image)");
	imshow("Epipolar line (first image)", input1);

	namedWindow("Epipolar line (second image)");
	imshow("Epipolar line (second image)", input2);
	

	//Mat A = cal_A(srcPoints, dstPoints, (int)srcPoints.size(), (int)srcPoints.size());
	Mat w, u, vt;
	//SVD::compute(A, w, u, vt);
	Mat h = vt;
	std::cout << h << std::endl;
	waitKey(0);

	return 0;
}

void drawlines(Mat& img, vector<Point3f>& lines, vector<Point2f>& img_pt)
{
	double r1, r2, r3;
	int img_width = img.size().width;	

	for (int i = 0; i < lines.size(); i+=20)
	{
		r1 = lines[i].x;
		r2 = lines[i].y;
		r3 = lines[i].z;

		// p1: leftmost point of line
		// p2: rightmost point of line

		
		Point p1, p2;
		
		p1 = Point(0,(double) r3/ (-r2));
		p2 = Point((img_width-1), (double) (r1 * (img_width) + r3) / (- r2)); //

		line(img, p1, p2, Scalar(0, 0, 255));
//		circle(img, Point(img_pt[i].x, img_pt[i].y), 2, Scalar(255, 255, 0));
	}
	
}

/**
* Calculate euclid distance
*/
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}

	return sqrt(sum);
}

/**
* Find the index of nearest neighbor point from keypoints.
*/
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors,int second) {
	int neighbor = -1;
	int sec = -1;
	double minDist = 1e6;
	double eDistance = 0.0;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor
		eDistance = euclidDistance(vec, v);

		if (minDist > eDistance) {
			sec = neighbor; // 수정되기 이전 값 저장
			minDist = eDistance;
			neighbor = i;
		}
	}


	if (second == 1) return sec;
	else return neighbor;
}

/**
* Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1, vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) { // image1 의 descriptors
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i);

		int nn = nearestNeighbor(desc1, keypoints2, descriptors2, 0);
		int l = nearestNeighbor(descriptors2.row(nn), keypoints1, descriptors1, 0);
		int secn = nearestNeighbor(desc1, keypoints2, descriptors2, 1);

		// Refine matching points using ratio_based thresholding
		if (!ratio_threshold) {

			float dist1 = euclidDistance(desc1, descriptors2.row(nn));
			float dist2 = euclidDistance(desc1, descriptors2.row(secn));

			float r = (float)dist1 / (float)dist2;
			if (r > RATIO_THR) continue;
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
			
			//	Fill the code
			if (i != l) continue;


		}

		KeyPoint pt2 = keypoints2[nn];
		srcPoints.push_back(pt1.pt);
		dstPoints.push_back(pt2.pt);
	}
}

void type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	printf("Matrix: %s \n", r.c_str());
}
template <typename T>
Mat cal_A(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points, int k) {

	Mat M(2 * number_of_points, 8, CV_32F, Scalar(0));
	Mat h(2 * number_of_points, 1, CV_32F);
	
	int numOfP = number_of_points;
	// initialize matrix
	/*
	int arr[4] = { 0,0,0 };
	int bCheckExistOfNum[500] = { 0 };
	for (int i = 0; i < number_of_points; i++) {
		bCheckExistOfNum[i] = 0;
	}


	for (int i = 0; i < k;i++) {
		int num = rand() % number_of_points;
		printf("random number %d\n", num);
		if (bCheckExistOfNum[num] == 0) {
			bCheckExistOfNum[num] = 1;
			arr[i] = num;
		}
		else i--;
	}
	*/
	for (int i = 0; i < k; i++) {
		Point2f pt1 = srcPoints[arr[i]];
		Point2f pt2 = dstPoints[arr[i]];
		M.at<T>(2 * i, 0) = pt1.y;		M.at<T>(2 * i, 1) = pt1.x;		M.at<T>(2 * i, 2) = 1;  M.at<T>(2 * i, 6) = -pt2.y*pt1.x;		M.at<T>(2 * i, 7) = -pt2.y*pt1.y;		M.at<T>(2 * i, 8) = -pt2.y;
		M.at<T>(2 * i + 1, 3) = pt1.y;		M.at<T>(2 * i + 1, 4) = pt1.x;		M.at<T>(2 * i + 1, 5) = 1;	M.at<T>(2 * i + 1, 6) = -pt2.x*pt1.y;		M.at<T>(2 * i + 1, 7) = -pt2.x*pt1.x;		M.at<T>(2 * i + 1, 8) = -pt2.x;
		
	}
	printf("!!\n");
	// (M^T * M)^(−1) * M^T * b ( * : Matrix multiplication)

	return M;
}
