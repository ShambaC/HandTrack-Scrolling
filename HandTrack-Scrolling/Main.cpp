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

		if (waitKey(1) > 0) {
			break;
		}
	}

	return 0;
}