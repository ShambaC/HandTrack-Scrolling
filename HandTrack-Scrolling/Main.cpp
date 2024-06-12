#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

int main() {
	cout << "OpenCV version is: " << CV_VERSION;

	Mat frame;

	VideoCapture cap;

	cap.open(0);

	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}

	Scalar lbound(39, 74, 23);
	Scalar ubound(349, 36, 35);

	while (true) {
		cap.read(frame);

		Mat hsvFrame;

		cv::cuda::GpuMat hsvGpu;
		//hsvGpu.upload(hsvFrame);
		cv::cuda::GpuMat frameGPU;
		frameGPU.upload(frame);
		cv::cuda::cvtColor(frameGPU, hsvGpu, COLOR_BGR2HSV);

		//Mat mask;
		//cv::cuda::inRange(hsvFrame, lbound, ubound, mask);

		imshow("Video Feed", frame);
		imshow("HSV", hsvGpu.download());
		//imshow("Mask", mask);

		if (waitKey(1) > 0) {
			break;
		}
	}

	return 0;
}