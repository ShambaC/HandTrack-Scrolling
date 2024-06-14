#include <iostream>
#include <algorithm>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

int main() {
	cout << "OpenCV version is: " << CV_VERSION << endl;

	Mat frame, hsvFrame, maskFrame;

	cv::cuda::GpuMat hsvGPU, frameGPU, maskGPU;

	VideoCapture cap;

	cap.open(0);

	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}

	Scalar lbound(0, 130, 200);
	Scalar ubound(30, 255, 255);

	int prev_x = NULL, prev_y = NULL;
	int scrollSensitivity = 50;

	while (true) {
		cap.read(frame);

		frameGPU.upload(frame);
		cv::cuda::cvtColor(frameGPU, hsvGPU, COLOR_BGR2HSV);
		hsvGPU.download(hsvFrame);

		cv::cuda::inRange(hsvGPU, lbound, ubound, maskGPU);
		maskGPU.download(maskFrame);

		imshow("Video Feed", frame);
		imshow("HSV", hsvFrame);
		imshow("Mask", maskFrame);

		vector<Mat> contours;

		findContours(maskFrame, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

		if (contours.size() > 0) {
			Mat maxContour = * max_element(contours.begin(),
									contours.end(),
									[](const auto& a, const auto& b) {
										return contourArea(a) < contourArea(b);
									});

			cv::cuda::GpuMat maxContourGPU;
			maxContourGPU.upload(maxContour);

			int cx, cy;

			Moments M = cv::cuda::moments(maxContourGPU);
			if (M.m00 != 0) {
				cx = (int)(M.m10 / M.m00);
				cy = (int)(M.m01 / M.m00);
				if (prev_y == NULL) {
					prev_y = cy;
				}

				if (abs(cy - prev_y) > scrollSensitivity) {
					int scroll_amount = prev_y - cy;

					mouse_event(MOUSEEVENTF_WHEEL, NULL, NULL, scroll_amount, NULL);

					prev_y = cy;
				}

			}
		}

		if (waitKey(1) > 0) {
			break;
		}
	}

	return 0;
}