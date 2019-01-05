#include "hist_func.h"

void hist_match(Mat &input, Mat &refInput, Mat &matched, G *trans_func, float *CDF, float *CDF_ref);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat refInput = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat refInput_gray;

	float difference;
	float min = 0;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to  Grayscale
	cvtColor(refInput, refInput_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

	Mat matched = input_gray.clone();
	Mat referenced = refInput_gray.clone();


	// PDF or transfer function txt files
	FILE *f_PDF; // histogram of the input image
	FILE *f_PDF_ref;
	FILE *f_matched_PDF_gray; //histogram of the output image
	FILE *f_trans_func_match; // transfer function txt
	
	fopen_s(&f_PDF, "PDF.txt", "w+");
	fopen_s(&f_PDF_ref, "PDF_ref.txt", "w+");
	
	fopen_s(&f_matched_PDF_gray, "matched_PDF_gray.txt", "w+");
	fopen_s(&f_trans_func_match, "trans_func_match.txt", "w+");

	float *PDF = cal_PDF(input_gray);	// PDF of Input image(Grayscale) : [L]  hist_func.h 에 정의된 함수의 리턴 값 가르킴
	float *CDF = cal_CDF(input_gray);	// CDF of Input image(Grayscale) : [L]
	float *PDF_ref = cal_PDF(refInput_gray); //PDF of reference image(Grayscale) : [L]
	float *CDF_ref = cal_CDF(refInput_gray);

	G trans_func_match[L] = { 0 };			// transfer function

	hist_match(input_gray, refInput_gray, matched, trans_func_match, CDF, CDF_ref);      // histogram equalization on grayscale image
	float *matched_PDF_gray = cal_PDF(matched);									// equalized PDF (grayscale)

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_PDF_ref, "%d\t%f\n", i, PDF_ref[i]);
		fprintf(f_matched_PDF_gray, "%d\t%f\n", i, matched_PDF_gray[i]);
		
		

		// write transfer functions
		fprintf(f_trans_func_match, "%d\t%d\n", i, trans_func_match[i]);
	}

	// memory release
	free(PDF);
	free(CDF);
	fclose(f_PDF);
	fclose(f_PDF_ref);
	fclose(f_matched_PDF_gray);
	fclose(f_trans_func_match);

	////////////////////// Show each image ///////////////////////

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	namedWindow("Grayscale of reference", WINDOW_AUTOSIZE);
	imshow("Grayscale of reference", refInput_gray);

	namedWindow("Matched", WINDOW_AUTOSIZE);
	imshow("Matched", matched);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization
void hist_match(Mat &input, Mat &refInput, Mat &matched, G *trans_func, float *CDF, float *CDF_ref) {

	float diff;
	float min;
	G tr[L] = { 0 };
	G gz[L] = { 0 };
	G inverse_hist[L] = { 0 };
	// compute transfer function G(z) and inverse_hist  of referenced image
	for (int i = 0; i < L; i++) {
		gz[i] = (G)(L - 1) * CDF_ref[i]; // equalize reference image
		
		}
	// compute inverse_hist
	for (int i = 0; i < L;i++) {
		diff = fabs(i - gz[0]);
		min = 0;
		for (int j = 0;j < L; j++) {
			if (fabs(i - gz[j]) < diff) {
				diff = fabs(i - gz[j]);
				min = j;

			}
			inverse_hist[i] = min;
		}
	}
	
	// compute transfer function T(r)
	for (int i = 0; i < L; i++)
		tr[i] = (G)((L - 1) * CDF[i]);

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			matched.at<G>(i, j) = inverse_hist[tr[input.at<G>(i, j)]];
	// 
	for (int i = 0; i < L;i++) { // trans_func 에 최종 transfer 변환 공식 넣어줌
		int j = tr[i];
		trans_func[i] = inverse_hist[j];
	}
}