#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#define M_PI	3.14159265358979323846

Mat convertToGreyScale(const Mat& src_image)
{
	Mat dest_image;

	int rows = src_image.rows;
	int cols = src_image.cols;

	dest_image.create(src_image.size(), CV_8UC1);

	for (int row = 0; row < rows; ++row)
	{
		const uchar* row_ptr = src_image.ptr<uchar>(row);
		uchar* dest_pixel_ptr = dest_image.ptr<uchar>(row);

		for (int col = 0; col < cols; col++)
		{
			dest_pixel_ptr[col] = (uchar)(row_ptr[0] * 0.114f + row_ptr[1] * 0.587f + row_ptr[2] * 0.299f);
			row_ptr += 3;
		}
	}

	return dest_image;
}

Mat getGaussianKernl(const int hsize, const double sigma)
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
			gaussian_kernel.at<double>(i, j) /= total;
		}
	}

	return gaussian_kernel;
}

Mat convolution(const Mat& src_image, const Mat& filter)
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

Mat sobelOperator(const Mat& src_image, const Mat1s& sobel_kernel, const int radius)
{
	Mat1f filtered_image(src_image.rows, src_image.cols);
	Mat1b border_image;
	copyMakeBorder(src_image, border_image, radius, radius, radius, radius, BORDER_REFLECT101);

	for (int i = radius; i < border_image.rows - radius; i++)
	{
		for (int j = radius; j < border_image.cols - radius; j++)
		{
			short sum = 0;
			for (int k = -radius; k <= radius; k++)
			{
				for (int l = -radius; l <= radius; l++)
				{
					sum += border_image(i + k, j + l) * sobel_kernel(k + radius, l + radius);
				}
			}
			filtered_image(i - radius, j - radius) = sum;
		}
	}

	return filtered_image;
}

Mat nonMaximumSuppression(const Mat& src_image, const Mat& src_sobel_x_gradient, const Mat& src_sobel_y_gradient)
{
	float theta;
	Mat non_maximal_suppressed_image = Mat(src_image.rows, src_image.cols, CV_8UC1);

	for (int i = 1; i < src_sobel_x_gradient.rows - 1; i++)
	{
		for (int j = 1; j < src_sobel_y_gradient.cols - 1; j++)
		{
			theta = atan2(src_sobel_y_gradient.at<uchar>(i, j), src_sobel_x_gradient.at<uchar>(i, j)) * (180 / M_PI);
			non_maximal_suppressed_image.at<uchar>(i - 1, j - 1) = src_image.at<uchar>(i, j);

			if (((-22.5 < theta) && (theta <= 22.5)) || ((157.5 < theta) && (theta <= -157.5)))
			{
				if ((src_image.at<uchar>(i, j) < src_image.at<uchar>(i, j + 1)) || (src_image.at<uchar>(i, j) < src_image.at<uchar>(i, j - 1)))
					non_maximal_suppressed_image.at<uchar>(i - 1, j - 1) = 0;
			}

			if (((-112.5 < theta) && (theta <= -67.5)) || ((67.5 < theta) && (theta <= 112.5)))
			{
				if ((src_image.at<uchar>(i, j) < src_image.at<uchar>(i + 1, j)) || (src_image.at<uchar>(i, j) < src_image.at<uchar>(i - 1, j)))
					non_maximal_suppressed_image.at<uchar>(i - 1, j - 1) = 0;
			}

			if (((-67.5 < theta) && (theta <= -22.5)) || ((112.5 < theta) && (theta <= 157.5)))
			{
				if ((src_image.at<uchar>(i, j) < src_image.at<uchar>(i - 1, j + 1)) || (src_image.at<uchar>(i, j) < src_image.at<uchar>(i + 1, j - 1)))
					non_maximal_suppressed_image.at<uchar>(i - 1, j - 1) = 0;
			}

			if (((-157.5 < theta) && (theta <= -112.5)) || ((22.5 < theta) && (theta <= 67.5)))
			{
				if ((src_image.at<uchar>(i, j) < src_image.at<uchar>(i + 1, j + 1)) || (src_image.at<uchar>(i, j) < src_image.at<uchar>(i - 1, j - 1)))
					non_maximal_suppressed_image.at<uchar>(i - 1, j - 1) = 0;
			}
		}

	}
	return non_maximal_suppressed_image;
}

Mat formulateGradientOrientation(const Mat& magnitude, Mat& vector_angles, double threshold = 1.0)
{
	Mat gradient_orientation_image = Mat::zeros(vector_angles.size(), CV_8UC3);

	for (int i = 0; i < magnitude.rows * magnitude.cols; i++)
	{
		float* magnitude_pixel = reinterpret_cast<float*>(magnitude.data + i * sizeof(float));
		if (*magnitude_pixel > threshold)
		{
			float* oriPixel = reinterpret_cast<float*>(vector_angles.data + i * sizeof(float));
			if (*oriPixel < 90.0)
				gradient_orientation_image.at<Vec3b>(Point(i)) = Vec3b(220, 220, 220);
			else if (*oriPixel >= 90.0 && *oriPixel < 180.0)
				gradient_orientation_image.at<Vec3b>(Point(i)) = Vec3b(165, 165, 165);
			else if (*oriPixel >= 180.0 && *oriPixel < 270.0)
				gradient_orientation_image.at<Vec3b>(Point(i)) = Vec3b(110, 110, 110);
			else if (*oriPixel >= 270.0 && *oriPixel < 360.0)
				gradient_orientation_image.at<Vec3b>(Point(i)) = Vec3b(55, 55, 55);
		}
	}
	return gradient_orientation_image;
}

Mat perfectEdgeFilter(const Mat& src_image)
{
	Mat gray_image;
	Mat dest_image;

	std::vector<Mat> channels;
	Mat hsv;
	cvtColor(src_image, hsv, CV_RGB2HSV);
	split(hsv, channels);
	gray_image = channels[0];

	Canny(src_image, dest_image, 1, 350);

	return dest_image;
}

void myEdgeFilter(const Mat& original_image, const int sigma)
{
	// Display Original Image
	namedWindow("Original");
	imshow("Original", original_image);

	Mat1b grey_image = convertToGreyScale(original_image);

	// Display Greyscale Image
	namedWindow("Greyscale");
	imshow("Greyscale", grey_image);

	double hsize = 2 * ceil(3 * sigma) + 1;

	Mat gaussian_filter = getGaussianKernl(hsize, sigma);

	Mat smoothed_image = convolution(grey_image, gaussian_filter);

	// Display Gaussian Blur Image
	namedWindow("Gaussian Blur");
	imshow("Gaussian Blur", smoothed_image);

	// Sobel Kernels (X-Direction & Y-Direction)
	Mat1s sobel_x_kernel = (Mat1s(3, 3) << -1, 0, +1, -2, 0, +2, -1, 0, +1);
	Mat1s sobel_y_kernel = (Mat1s(3, 3) << -1, -2, -1, 0, 0, 0, +1, +2, +1);

	// Pass Over Sobel Filter in X-Direction and Y-Direction
	Mat sobel_x_image = sobelOperator(smoothed_image, sobel_x_kernel, 1);
	Mat sobel_y_image = sobelOperator(smoothed_image, sobel_y_kernel, 1);
	Mat1b sobel_final_image;
	absdiff(sobel_x_image, sobel_y_image, sobel_final_image);

	// Display Gradient Magnitude Image (Without Non-Maximum Suppression)
	namedWindow("Gradient Magnitude");
	imshow("Gradient Magnitude", sobel_final_image);

	Mat non_maximal_image = nonMaximumSuppression(sobel_final_image, sobel_x_image, sobel_y_image);

	//Compute Gradient Magnitude & Array of Angles - Running Out Of Time
	Mat1f magnitude_matrix;
	Mat1f vector_angles;
	magnitude(sobel_x_image, sobel_y_image, magnitude_matrix);
	phase(sobel_x_image, sobel_y_image, vector_angles, true);

	Mat gradient_orientation_image = formulateGradientOrientation(magnitude_matrix, vector_angles, 1.0);

	// Display Gradient Magnitude Image (With Non-Maximum Suppression)
	namedWindow("Gradient Magnitude (With NMS)");
	imshow("Gradient Magnitude (With NMS)", non_maximal_image);

	// Display Gradient Orientation Image
	namedWindow("Gradient Orientation");
	imshow("Gradient Orientation", gradient_orientation_image);
}

int main(int argc, char** argv)
{
	Mat original_image = imread("img0.jpg");

	int sigma = 1;

	myEdgeFilter(original_image, sigma);

	// Canny Edge Detection OpenCV Library Solution
	// Mat perfect_image = perfectEdgeFilter(original_image);
	// namedWindow("Perfect");
	// imshow("Perfect", perfect_image);

	waitKey(0);

	return 0;
}