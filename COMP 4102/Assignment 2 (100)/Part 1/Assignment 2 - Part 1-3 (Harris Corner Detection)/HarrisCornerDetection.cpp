#include "opencv2/highgui.hpp"

#define EIGENVALUE		500000
#define APETURE_SIZE	3
#define BLOCK_SIZE		3
#define MAX_CORNERS		100000

void harrisCornerDetection(IplImage* input_image, IplImage* output_image)
{
	CvMat* x_derivative = cvCreateMat(input_image->height, input_image->width, CV_16SC1);
	CvMat* y_derivative = cvCreateMat(input_image->height, input_image->width, CV_16SC1);
	CvMat* x_derivative_prime = cvCreateMat(input_image->height, input_image->width, CV_64FC1);
	CvMat* y_derivative_prime = cvCreateMat(input_image->height, input_image->width, CV_64FC1);
	CvMat* minimum_matrix = cvCreateMat(input_image->height, input_image->width, CV_64FC1);
	CvMat* maximum_matrix = cvCreateMat(input_image->height, input_image->width, CV_64FC1);
	CvMat* r_matrix	= cvCreateMat(input_image->height, input_image->width, CV_64FC1);

	cvSobel(input_image, x_derivative, 1, 0, APETURE_SIZE);
	cvSobel(input_image, y_derivative, 0, 1, APETURE_SIZE);

	cvConvertScale(x_derivative, x_derivative_prime);
	cvConvertScale(y_derivative, y_derivative_prime);

	CvPoint3D32f* corners = (CvPoint3D32f*)cvAlloc(MAX_CORNERS * sizeof(corners));

	int corner_count = 0;
	CvMat* new_Matrix = cvCreateMat(2, 2, CV_64FC1);

	int x_distance = 0;
	while (x_distance < input_image->height)
	{
		int y_distance = 0;
		while (y_distance < input_image->width)
		{
			int new_x_value = 0;
			int new_y_value = 0;
			int new_xy_value = 0;

			int i = x_distance - floor((double)BLOCK_SIZE / 2);
			while (i <= x_distance + floor((double)BLOCK_SIZE / 2))
			{
				int j = y_distance - floor((double)BLOCK_SIZE / 2);
				while (j <= y_distance + floor((double)BLOCK_SIZE / 2))
				{
					if (i < 0 || j < 0 || i >= input_image->height || j >= input_image->width)
					{
						j++;
						continue;
					}
					new_x_value += pow(cvmGet(x_derivative_prime, i, j), 2);
					new_y_value += pow(cvmGet(y_derivative_prime, i, j), 2);
					new_xy_value += cvmGet(x_derivative_prime, i, j) * cvmGet(y_derivative_prime, i, j);
					j++;
				}
				i++;
			}

			cvmSet(new_Matrix, 0, 0, new_x_value);
			cvmSet(new_Matrix, 0, 1, new_xy_value);
			cvmSet(new_Matrix, 1, 0, new_xy_value);
			cvmSet(new_Matrix, 1, 1, new_y_value);

			CvMat* vectors = cvCreateMat(2, 2, CV_64FC1);
			CvMat* values = cvCreateMat(1, 2, CV_64FC1);

			cvEigenVV(new_Matrix, vectors, values, -1, -1);

			double determinant = (cvmGet(values, 0, 0) * cvmGet(values, 0, 1));
			double trace = (cvmGet(values, 0, 0) + cvmGet(values, 0, 1));
			double Ri = (determinant - 0.04 * pow(trace, 2));

			cvmSet(r_matrix, x_distance, y_distance, Ri);
			if (cvmGet(values, 0, 0) <= cvmGet(values, 0, 1))
			{
				cvmSet(minimum_matrix, x_distance, y_distance, cvmGet(values, 0, 0));
				cvmSet(maximum_matrix, x_distance, y_distance, cvmGet(values, 0, 1));
			}
			else
			{
				cvmSet(minimum_matrix, x_distance, y_distance, cvmGet(values, 0, 1));
				cvmSet(maximum_matrix, x_distance, y_distance, cvmGet(values, 0, 0));
			}

			if (cvmGet(values, 0, 0) > EIGENVALUE && cvmGet(values, 0, 1) > EIGENVALUE && Ri >= EIGENVALUE)
			{
				CvPoint3D32f point;
				point.x = x_distance;
				point.y = y_distance;
				point.z = 0;
				corners[corner_count++] = point;
			}
			y_distance++;
		}
		x_distance++;
	}

	CvScalar point_color = cvScalar(0, 0, 255, 0);
	for (int i = 0; i < corner_count; i++) {
		CvPoint edge_marker = cvPoint(corners[i].y, corners[i].x);
		cvCircle(output_image, edge_marker, 1, point_color, 1, 8, 0);
	}

	cvNormalize(minimum_matrix, minimum_matrix, 255, 0);
	cvShowImage("Minimum Image Output", minimum_matrix);
	cvNormalize(maximum_matrix, maximum_matrix, 255, 0);
	cvShowImage("Maximum Image Output ", maximum_matrix);
	cvNormalize(r_matrix, r_matrix, 255, 0);
	cvShowImage("R Image Output", r_matrix);
	cvShowImage("Final Image Output", output_image);
}

int main(int argc, char** argv)
{
	IplImage* original_image = cvLoadImage("box_in_scene.png", 1);
	cvShowImage("Original Image", original_image);

	IplImage* grayscale_image = cvCreateImage(cvSize(original_image->width, original_image->height), IPL_DEPTH_8U, 1);
	cvCvtColor(original_image, grayscale_image, CV_BGR2GRAY);

	harrisCornerDetection(grayscale_image, original_image);

	cvWaitKey(0);

	return 0;
}