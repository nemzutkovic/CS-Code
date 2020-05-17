#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void updatePointVectors(vector<vector<DMatch>>* adjust_vector, vector<Point2f>* other_img_matches, vector<Point2f>* middle_img_matches, vector<KeyPoint>* other_point_detector, vector<KeyPoint>* middle_point_detector)
{
	for (size_t i = 0; i < adjust_vector->size(); i++) {
		DMatch first = (*adjust_vector)[i][0];
		float distance_1 = (*adjust_vector)[i][0].distance;
		float distance_2 = (*adjust_vector)[i][1].distance;

		if (distance_1 < 0.8 * distance_2) {
			other_img_matches->push_back((*other_point_detector)[first.queryIdx].pt);
			middle_img_matches->push_back((*middle_point_detector)[first.trainIdx].pt);
		}
	}
}

void stitchImage(Mat* img_to_warp, Mat* tmp_img, Mat* middle_img, Mat* final_image, Mat img_homography)
{
	warpPerspective(*img_to_warp, *tmp_img, img_homography, tmp_img->size());

	for (int i = 0; i < tmp_img->cols; ++i) {
		for (int j = 0; j < tmp_img->rows; ++j) {
			if (middle_img->at<uchar>(j, i) == 0) {
				final_image->at<uchar>(j, i) = tmp_img->at<uchar>(j, i);
			}
			else if (tmp_img->at<uchar>(j, i) == 0) {
				final_image->at<uchar>(j, i) = middle_img->at<uchar>(j, i);
			}
			else {
				final_image->at<uchar>(j, i) = (middle_img->at<uchar>(j, i) + tmp_img->at<uchar>(j, i)) / 2;
			}
		}
	}
}

void displayPanoramicImage(Mat* left_img, Mat* middle_img, Mat* right_img)
{
	Mat tmp_img = Mat(middle_img->rows, middle_img->cols, CV_8UC1);

	vector<KeyPoint> left_img_point_detector;
	vector<KeyPoint> middle_img_point_detector;
	vector<KeyPoint> right_img_point_detector;

	Mat left_img_descriptor;
	Mat middle_img_descriptor;
	Mat right_img_descriptor;

	Ptr<AKAZE> akaze_keypoint_detector = AKAZE::create();

	akaze_keypoint_detector->detectAndCompute(*left_img, noArray(), left_img_point_detector, left_img_descriptor);
	akaze_keypoint_detector->detectAndCompute(*middle_img, noArray(), middle_img_point_detector, middle_img_descriptor);
	akaze_keypoint_detector->detectAndCompute(*right_img, noArray(), right_img_point_detector, right_img_descriptor);

	BFMatcher descriptor_matcher(NORM_HAMMING);

	vector<vector<DMatch>> nn_matches1;
	vector<vector<DMatch>> nn_matches2;

	descriptor_matcher.knnMatch(left_img_descriptor, middle_img_descriptor, nn_matches1, 2);
	descriptor_matcher.knnMatch(right_img_descriptor, middle_img_descriptor, nn_matches2, 2);

	vector<Point2f> left_img_matches;
	vector<Point2f> middle_img_matches;
	vector<Point2f> right_img_matches;
	vector<Point2f> middle_img_matches2;

	updatePointVectors(&nn_matches1, &left_img_matches, &middle_img_matches, &left_img_point_detector, &middle_img_point_detector);
	updatePointVectors(&nn_matches2, &right_img_matches, &middle_img_matches2, &right_img_point_detector, &middle_img_point_detector);

	Mat left_img_homography = findHomography(left_img_matches, middle_img_matches, RANSAC);
	Mat right_img_homography = findHomography(right_img_matches, middle_img_matches2, RANSAC);

	Mat final_image = Mat(tmp_img.rows, tmp_img.cols, CV_8UC1);

	stitchImage(left_img, &tmp_img, middle_img, &final_image, left_img_homography);
	stitchImage(right_img, &tmp_img, &final_image, &final_image, right_img_homography);

	imshow("Stitched Image", final_image);
}

int main(void)
{
	Mat left_img   = imread("keble_a_half.bmp", IMREAD_GRAYSCALE);
	Mat middle_img = imread("keble_b_long.bmp", IMREAD_GRAYSCALE);
	Mat right_img  = imread("keble_c_half.bmp", IMREAD_GRAYSCALE);

	displayPanoramicImage(&left_img, &middle_img, &right_img);
	waitKey(0);

	return 0;
}