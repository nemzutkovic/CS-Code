#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void project3Dto2D() {

    Mat t_matrix = (Mat_<float>(3, 1) << -70, -95, -120);
    cout << "Given Camera Location: " << endl << " " << t_matrix << endl << endl;

    Mat r_matrix = (Mat_<float>(3, 4) << 1, 0, 0, -70, 0, 1, 0, -95, 0, 0, 1, -120);
    cout << "Given Identity Matrix with Camera Location: " << endl << " " << r_matrix << endl << endl;

    Mat intrinsic_matrix = (Mat_<float>(3, 3) << -500, 0, 320, 0, -500, 240, 0, 0, 1);
    cout << "Intrinsic Matrix: " << endl << intrinsic_matrix << endl << endl;

    Mat projection_matrix = intrinsic_matrix * r_matrix;
    cout << "3 X 4 Projection Matrix: " << endl << projection_matrix << endl << endl;

    Mat zero_rotation_matrix = (Mat_<float>(3, 1) << 0, 0, 0);

    Mat distortion_matrix = (Mat_<float>(4, 1) << 0, 0, 0, 0);

    // Declaring 2D Points
    vector<Point2f> two_d_points;

    // Declaring 3D Points
    vector<Point3f> three_d_points;

    three_d_points.push_back(Point3f(150, 200, 400));
    cout << "World Point Xw: " << three_d_points << endl << endl;

    projectPoints(three_d_points, zero_rotation_matrix, t_matrix, intrinsic_matrix, distortion_matrix, two_d_points);

    cout << "Projection of 3D to 2D: " << two_d_points << endl;
}

int main(int argc, const char* argv[]) {

    project3Dto2D();

    waitKey(0);
    return 0;
}