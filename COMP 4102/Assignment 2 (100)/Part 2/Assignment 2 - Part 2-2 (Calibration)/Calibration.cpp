#include "opencv2/opencv.hpp"

#define NUM_POINTS    10
#define RANGE         100.00

#define MAX_CAMERAS   100 
#define MAX_POINTS    3000

double projection[3][4] = {
0.902701, 0.051530, 0.427171, 10.000000,
0.182987, 0.852568, -0.489535, 15.000000,
-0.389418, 0.520070, 0.760184, 20.000000,
};

double intrinsic[3][3] = {
-1000.000000, 0.000000, 0.000000,
0.000000, -2000.000000, 0.000000,
0.000000, 0.000000, 1.000000,
};

double all_object_points[10][3] = {
71.0501, 51.3535, 30.3995,
1.4985, 9.1403, 36.4452,
14.7313, 16.5899, 98.8525,
44.5692, 11.9083, 0.4669,
0.8911, 37.7880, 53.1663,
0.1251, 56.3585, 19.3304,
80.8741, 58.5009, 47.9873,
35.0291, 89.5962, 82.2840,
74.6605, 17.4108, 85.8943,
57.1184, 60.1764, 60.7166,
};

void decomposeprojectionmatrix(CvMat* computed_projection_matrix, CvMat* rotation_matrix, CvMat* translation, CvMat* camera_matrix) {

	CvMat* projection_matrix = cvCreateMat(3, 4, CV_64F);

	float denominator = sqrt(pow(cvGetReal2D(computed_projection_matrix, 2, 0), 2) + pow(cvGetReal2D(computed_projection_matrix, 2, 1), 2) + pow(cvGetReal2D(computed_projection_matrix, 2, 2), 2));

	int row = 0, col = 0;
	while (row < 3)
	{
		while (col < 4)
		{
			cvSetReal2D(projection_matrix, row, col, cvGetReal2D(computed_projection_matrix, row, col) / denominator);
			col++;
		}
		col = 0;
		row++;
	}

	float num = (cvGetReal2D(computed_projection_matrix, 2, 3) < 0) ? -1 : 1;

	int m = 0;
	while (m < 3)
	{
		cvSetReal2D(rotation_matrix, 2, m, num * cvGetReal2D(projection_matrix, 2, m));
		m++;
	}

	float ox = ((cvGetReal2D(computed_projection_matrix, 0, 0) * cvGetReal2D(computed_projection_matrix, 2, 0)) + (cvGetReal2D(computed_projection_matrix, 0, 1) * cvGetReal2D(computed_projection_matrix, 2, 1)) + (cvGetReal2D(computed_projection_matrix, 0, 2) * cvGetReal2D(computed_projection_matrix, 2, 2))) * -1;
	float oy = ((cvGetReal2D(computed_projection_matrix, 1, 0) * cvGetReal2D(computed_projection_matrix, 2, 0)) + (cvGetReal2D(computed_projection_matrix, 1, 1) * cvGetReal2D(computed_projection_matrix, 2, 1)) + (cvGetReal2D(computed_projection_matrix, 1, 2) * cvGetReal2D(computed_projection_matrix, 2, 2))) * -1;
	float fx = round(sqrt((pow(cvGetReal2D(projection_matrix, 0, 0), 2) + pow(cvGetReal2D(projection_matrix, 0, 1), 2) + pow(cvGetReal2D(projection_matrix, 0, 2), 2)) - pow(ox, 2)));
	float fy = sqrt((pow(cvGetReal2D(projection_matrix, 1, 0), 2) + pow(cvGetReal2D(projection_matrix, 1, 1), 2) + pow(cvGetReal2D(projection_matrix, 1, 2), 2)) - pow(oy, 2));
	
	int n = 0;
	while (n < 3)
	{
		cvSetReal2D(rotation_matrix, 0, n, num * (ox * cvGetReal2D(projection_matrix, 2, n) - cvGetReal2D(projection_matrix, 0, n)) / fx);
		cvSetReal2D(rotation_matrix, 1, n, num * (oy * cvGetReal2D(projection_matrix, 2, n) - cvGetReal2D(projection_matrix, 1, n)) / fy);
		n++;
	}

	cvSetReal2D(translation, 0, 0, round((num * ((ox * (num * cvGetReal2D(projection_matrix, 2, 3))) - cvGetReal2D(projection_matrix, 0, 3)) / fx)));
	cvSetReal2D(translation, 1, 0, round((num * ((oy * (num * cvGetReal2D(projection_matrix, 2, 3))) - cvGetReal2D(projection_matrix, 1, 3)) / fy)));
	cvSetReal2D(translation, 2, 0, round(num * cvGetReal2D(projection_matrix, 2, 3)));

	cvSetReal2D(camera_matrix, 0, 0, -fx);
	cvSetReal2D(camera_matrix, 0, 1, 0);
	cvSetReal2D(camera_matrix, 0, 2, ox);
	cvSetReal2D(camera_matrix, 1, 0, 0);
	cvSetReal2D(camera_matrix, 1, 1, -fy);
	cvSetReal2D(camera_matrix, 1, 2, oy);
	cvSetReal2D(camera_matrix, 2, 0, 0);
	cvSetReal2D(camera_matrix, 2, 1, 0);
	cvSetReal2D(camera_matrix, 2, 2, 1);
}

void computeprojectionmatrix(CvMat* image_points, CvMat* object_points, CvMat* projection_matrix) {

	CvMat* eigenvectors = cvCreateMat(12, 12, CV_64F);
	CvMat* eigenvalues = cvCreateMat(12, 1, CV_64F);
	CvMat* source_matrix = cvCreateMat(NUM_POINTS * 2, 12, CV_64F);
	CvMat* destination_matrix = cvCreateMat(12, NUM_POINTS * 2, CV_64F);
	CvMat* final_matrix = cvCreateMat(12, 12, CV_64F);

	int point = 0, m = 0;
	while (m < (NUM_POINTS * 2))
	{
		float x = cvGetReal2D(object_points, point, 0);
		float y = cvGetReal2D(object_points, point, 1);
		float z = cvGetReal2D(object_points, point, 2);
		float xprime = cvGetReal2D(image_points, point, 0);
		float yprime = cvGetReal2D(image_points, point, 1);

		cvSetReal2D(source_matrix, m + 1, 0, 0);
		cvSetReal2D(source_matrix, m + 1, 1, 0);
		cvSetReal2D(source_matrix, m + 1, 2, 0);
		cvSetReal2D(source_matrix, m + 1, 3, 0);
		cvSetReal2D(source_matrix, m + 1, 4, x);
		cvSetReal2D(source_matrix, m + 1, 5, y);
		cvSetReal2D(source_matrix, m + 1, 6, z);
		cvSetReal2D(source_matrix, m + 1, 7, 1);
		cvSetReal2D(source_matrix, m + 1, 8, -(yprime * x));
		cvSetReal2D(source_matrix, m + 1, 9, -(yprime * y));
		cvSetReal2D(source_matrix, m + 1, 10, -(yprime * z));
		cvSetReal2D(source_matrix, m + 1, 11, -yprime);

		cvSetReal2D(source_matrix, m, 0, x);
		cvSetReal2D(source_matrix, m, 1, y);
		cvSetReal2D(source_matrix, m, 2, z);
		cvSetReal2D(source_matrix, m, 3, 1);
		cvSetReal2D(source_matrix, m, 4, 0);
		cvSetReal2D(source_matrix, m, 5, 0);
		cvSetReal2D(source_matrix, m, 6, 0);
		cvSetReal2D(source_matrix, m, 7, 0);
		cvSetReal2D(source_matrix, m, 8, -(xprime * x));
		cvSetReal2D(source_matrix, m, 9, -(xprime * y));
		cvSetReal2D(source_matrix, m, 10, -(xprime * z));
		cvSetReal2D(source_matrix, m, 11, -xprime);

		point++;
		m += 2;
	}

	cvTranspose(source_matrix, destination_matrix);
	cvMatMul(destination_matrix, source_matrix, final_matrix);
	cvEigenVV(final_matrix, eigenvectors, eigenvalues);

	int index = 0, row = 0, col = 0;
	while (row < 3)
	{
		while (col < 4)
		{
			cvSetReal2D(projection_matrix, row, col, cvGetReal2D(eigenvectors, 11, index++));
			col++;
		}
		col = 0;
		row++;
	}
}

int main() {
	CvMat* camera_matrix, * computed_camera_matrix;
	CvMat* rotation_matrix, * computed_rotation_matrix;
	CvMat* translation, * computed_translation;
	CvMat* image_points;
	CvMat* rot_vector;
	CvMat* object_points;
	CvMat* computed_projection_matrix;
	FILE* fp;

	object_points = cvCreateMat(NUM_POINTS, 3, CV_64F);
	image_points = cvCreateMat(NUM_POINTS, 2, CV_64F);

	rot_vector = cvCreateMat(3, 1, CV_64F);
	camera_matrix = cvCreateMat(3, 3, CV_64F);
	rotation_matrix = cvCreateMat(3, 3, CV_64F);
	translation = cvCreateMat(3, 1, CV_64F);

	computed_camera_matrix = cvCreateMat(3, 3, CV_64F);
	computed_rotation_matrix = cvCreateMat(3, 3, CV_64F);
	computed_translation = cvCreateMat(3, 1, CV_64F);
	computed_projection_matrix = cvCreateMat(3, 4, CV_64F);

	fp = fopen("assign2-Part2b.txt", "w");

	fprintf(fp, "Rotation matrix\n");
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			cvmSet(camera_matrix, i, j, intrinsic[i][j]);
			cvmSet(rotation_matrix, i, j, projection[i][j]);
		}
		fprintf(fp, "%f %f %f\n",
			cvmGet(rotation_matrix, i, 0), cvmGet(rotation_matrix, i, 1), cvmGet(rotation_matrix, i, 2));
	}
	for (int i = 0; i < 3; i++)
		cvmSet(translation, i, 0, projection[i][3]);

	fprintf(fp, "\nTranslation vector\n");
	fprintf(fp, "%f %f %f\n",
		cvmGet(translation, 0, 0), cvmGet(translation, 1, 0), cvmGet(translation, 2, 0));

	fprintf(fp, "\nCamera Calibration\n");
	for (int i = 0; i < 3; i++) {
		fprintf(fp, "%f %f %f\n",
			cvmGet(camera_matrix, i, 0), cvmGet(camera_matrix, i, 1), cvmGet(camera_matrix, i, 2));
	}

	fprintf(fp, "\n");
	for (int i = 0; i < NUM_POINTS; i++) {
		cvmSet(object_points, i, 0, all_object_points[i][0]);
		cvmSet(object_points, i, 1, all_object_points[i][1]);
		cvmSet(object_points, i, 2, all_object_points[i][2]);
		fprintf(fp, "Object point %d x %f y %f z %f\n",
			i, all_object_points[i][0], all_object_points[i][1], all_object_points[i][2]);
	}
	fprintf(fp, "\n");

	cvRodrigues2(rotation_matrix, rot_vector);
	cvProjectPoints2(object_points, rot_vector, translation, camera_matrix, NULL, image_points);

	for (int i = 0; i < NUM_POINTS; i++) {
		fprintf(fp, "Image point %d x %f y %f\n",
			i, cvmGet(image_points, i, 0), cvmGet(image_points, i, 1));
	}

	computeprojectionmatrix(image_points, object_points, computed_projection_matrix);
	decomposeprojectionmatrix(computed_projection_matrix, computed_rotation_matrix, computed_translation, computed_camera_matrix);

	fprintf(fp, "\nComputed Rotation matrix\n");
	for (int i = 0; i < 3; i++) {
		fprintf(fp, "%f %f %f\n",
			cvmGet(computed_rotation_matrix, i, 0), cvmGet(computed_rotation_matrix, i, 1), cvmGet(computed_rotation_matrix, i, 2));
	}

	fprintf(fp, "\nComputed Translation vector\n");
	fprintf(fp, "%f %f %f\n",
		cvmGet(computed_translation, 0, 0), cvmGet(computed_translation, 1, 0), cvmGet(computed_translation, 2, 0));

	fprintf(fp, "\nComputed Camera Calibration\n");
	for (int i = 0; i < 3; i++) {
		fprintf(fp, "%f %f %f\n",
			cvmGet(computed_camera_matrix, i, 0), cvmGet(computed_camera_matrix, i, 1), cvmGet(computed_camera_matrix, i, 2));
	}

	fclose(fp);
	return 0;
}