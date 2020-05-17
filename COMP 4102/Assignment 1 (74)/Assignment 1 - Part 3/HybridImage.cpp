#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#define M_PI	3.14159265358979323846

Mat gaussian_blur_kernel_2d(const int hsize, const double sigma)
{
	Mat gaussian_kernel(hsize, hsize, CV_64F);
	double total = 0.0;

	for (int i = 0; i < gaussian_kernel.rows; i++) {
		for (int j = 0; j < gaussian_kernel.cols; j++) {
			gaussian_kernel.at<double>(i, j) = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
			total += gaussian_kernel.at<double>(j);
		}
	}

	for (int i = 0; i < gaussian_kernel.rows; i++) {
		for (int j = 0; j < gaussian_kernel.cols; j++) {
			gaussian_kernel.at<double>(i, j) += total;
		}
	}

	return gaussian_kernel;
}

Mat convolve_2d(const Mat& src_image, const Mat& filter)
{
	Mat filtered_image(src_image.rows, src_image.cols, CV_8UC1);

	for (int i = 0; i < src_image.rows; i++) {
		for (int j = 0; j < src_image.cols; j++) {
			float pixel = 0;
			for (int k = 0; k < filter.rows; k++) {
				for (int l = 0; l < filter.cols; l++) {
					pixel += (src_image.at<uchar>(i + k, j + l) * filter.at<double>(k, l)); // Debug-Mode Issue Here
				}
			}
			filtered_image.at<uchar>(i, j) = pixel / sum(filter)[0];
		}
	}

	return filtered_image;
}

void hybridFilter(const Mat& first_image, const Mat& second_image, const int sigma)
{
	// Display First Image
	namedWindow("Cat");
	imshow("Cat", first_image);

	// Display Second Image
	namedWindow("Dog");
	imshow("Dog", second_image);

	double hsize = 2 * ceil(3 * sigma) + 1;

	Mat gaussian_filter = gaussian_blur_kernel_2d(hsize, sigma);

	Mat cat_smoothed_image = convolve_2d(first_image, gaussian_filter);
	Mat dog_smoothed_image = convolve_2d(second_image, gaussian_filter);

	// Display Blurred Cat Image
	namedWindow("Cat Blurred");
	imshow("Cat Blurred", cat_smoothed_image);

	// Display Blurred Dog Image
	namedWindow("Dog Blurred");
	imshow("Dog Blurred", dog_smoothed_image);
}

int main(int argc, char** argv)
{
	Mat first_image = imread("cat2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat second_image = imread("littledog.png", CV_LOAD_IMAGE_GRAYSCALE);

	int sigma = 1;

	hybridFilter(first_image, second_image, sigma);

	waitKey(0);

	return 0;
}