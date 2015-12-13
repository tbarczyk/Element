#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#ifndef PI
#define PI 3.1415
#endif
struct calibrationResult{
	std::string msg;
	std::vector<cv::Mat> rvecs;
	std::vector<cv::Mat> tvecs;
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	bool ok;
};
static class FilesCalibration{
public:
	static calibrationResult StartFilesCalibration();
};
