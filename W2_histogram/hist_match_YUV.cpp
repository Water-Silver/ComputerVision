#include "hist_func.h"

void hist_match_YUV(Mat &input, Mat &refInput, Mat &matched,Mat &matched_ref, G *trans_func, float *CDF, float *CDF_ref);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat refInput = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	
	Mat equalized_YUV;
	Mat equalized_YUV_ref;
	Mat matched_YUV;

	cvtColor(input, equalized_YUV, CV_RGB2YUV);	// RGB -> YUV
	cvtColor(refInput, equalized_YUV_ref, CV_RGB2YUV);

												// split each channel(Y, U, V)
	Mat channels[3];
	split(equalized_YUV, channels);
	Mat Y = channels[0];						// U = channels[1], V = channels[2]

	Mat channels_ref[3];
	split(equalized_YUV_ref, channels_ref);
	Mat Y_ref = channels_ref[0];

												// PDF or transfer function txt files
	
	FILE *f_matched_PDF_YUV, *f_PDF_RGB, *f_PDF_RGB_ref;

	FILE *f_trans_func_YUV; // transfer function 적을 파일 이름

	float **PDF_RGB = cal_PDF_RGB(input);		// PDF of Input image(RGB) : [L][3]
	float **PDF_RGB_ref = cal_PDF_RGB(refInput);
	float *CDF_YUV = cal_CDF(Y);				// CDF of Y channel image
	float *CDF_YUV_ref = cal_CDF(Y_ref);

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_PDF_RGB_ref, "PDF_RGB_ref.txt", "w+");
	fopen_s(&f_matched_PDF_YUV, "matched_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_YUV, "trans_func_eq_YUV.txt", "w+");

	G trans_func_match_YUV[L] = { 0 };			// transfer function

											// histogram equalization on Y channel
	hist_match_YUV(Y, Y_ref, channels[0],channels_ref[0], trans_func_match_YUV, CDF_YUV, CDF_YUV_ref);

	// merge Y, U, V channels
	merge(channels, 3, matched_YUV); // channels[0] 에 들어간 matched value

	// YUV -> RGB (use "CV_YUV2RGB" flag)
	cvtColor(matched_YUV, matched_YUV, CV_YUV2RGB);

	// equalized PDF (YUV)
	float **matched_PDF_YUV = cal_PDF_RGB(matched_YUV);

	for (int i = 0; i < L; i++) {
		// write PDF
		for (int j = 0;j < 3;j++) {
			fprintf(f_PDF_RGB, "%d %d \t%f\n", i, j, PDF_RGB[i][j]);
			fprintf(f_PDF_RGB_ref, "%d %d \t%f\n", i, j, PDF_RGB_ref[i][j]);
			fprintf(f_matched_PDF_YUV, "%d %d\t%f\n", i, j, matched_PDF_YUV[i][j]);

		}
		// write transfer functions
		fprintf(f_trans_func_YUV, "%d \t%d\n", i, trans_func_match_YUV[i]);

	}

	// memory release
	free(PDF_RGB);
	free(CDF_YUV);
	fclose(f_PDF_RGB);
	fclose(f_PDF_RGB_ref);
	fclose(f_matched_PDF_YUV);
	fclose(f_trans_func_YUV);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Reference", WINDOW_AUTOSIZE);
	imshow("Reference", refInput);

	namedWindow("Matched_YUV", WINDOW_AUTOSIZE);
	imshow("Matched_YUV", matched_YUV);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram match for channel Y
void hist_match_YUV(Mat &input, Mat &refInput, Mat &matched, Mat &matched_ref, G *trans_func, float *CDF, float *CDF_ref){

	float diff;
	float min;
	G tr[L] = { 0 };
	G gz[L] = { 0 };
	G inverse_hist[L] = { 0 };

	// compute transfer function G(z) and inverse_hist  of referenced image
	for (int i = 0; i < L; i++) {
		gz[i] = (G)(L - 1) * CDF_ref[i]; // equalize
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
	for (int i = 0; i < L;i++) {
		int j = tr[i];
		trans_func[i] = inverse_hist[j];
	}
}