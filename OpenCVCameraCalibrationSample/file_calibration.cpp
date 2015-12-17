

#include "stdafx.h"
#define FCABLIB 1;
#include "file_calibration.h"
#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <cmath>
#include "Element.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/ocl/ocl.hpp"
#pragma comment (lib,"OpenCL.lib")
#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

#ifndef _CRT_SECURE_NO_WARNINGS
# define _CRT_SECURE_NO_WARNINGS
#endif
#define HAVE_CUFFT 1
using namespace cv;
using namespace std;

static void help()
{
	cout << "This is a camera calibration sample." << endl
		<< "Usage: calibration configurationFile" << endl
		<< "Near the sample file you'll find the configuration file, which has detailed help of "
		"how to edit it.  It may be any OpenCV supported file format XML/YAML." << endl;
	getchar();
};



class Settings
{
public:
	Settings() : goodInput(false) {}
	enum Pattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
	enum InputType { INVALID, CAMERA, VIDEO_FILE, IMAGE_LIST };

	void write(FileStorage& fs) const                        //Write serialization for this class
	{
		fs << "{" << "BoardSize_Width" << boardSize.width
			<< "BoardSize_Height" << boardSize.height
			<< "Square_Size" << squareSize
			<< "Calibrate_Pattern" << patternToUse
			<< "Calibrate_NrOfFrameToUse" << nrFrames
			<< "Calibrate_FixAspectRatio" << aspectRatio
			<< "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
			<< "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint

			<< "Write_DetectedFeaturePoints" << bwritePoints
			<< "Write_extrinsicParameters" << bwriteExtrinsics
			<< "Write_outputFileName" << outputFileName

			<< "Show_UndistortedImage" << showUndistorsed

			<< "Input_FlipAroundHorizontalAxis" << flipVertical
			<< "Input_Delay" << delay
			<< "Input" << input
			<< "}";
	}
	void read(const FileNode& node)                          //Read serialization for this class
	{
		node["BoardSize_Width"] >> boardSize.width;
		node["BoardSize_Height"] >> boardSize.height;
		node["Calibrate_Pattern"] >> patternToUse;
		node["Square_Size"] >> squareSize;
		node["Calibrate_NrOfFrameToUse"] >> nrFrames;
		node["Calibrate_FixAspectRatio"] >> aspectRatio;
		node["Write_DetectedFeaturePoints"] >> bwritePoints;
		node["Write_extrinsicParameters"] >> bwriteExtrinsics;
		node["Write_outputFileName"] >> outputFileName;
		node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
		node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
		node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
		node["Show_UndistortedImage"] >> showUndistorsed;
		node["Input"] >> input;
		node["Input_Delay"] >> delay;
		interprate();
	}
	void interprate()
	{
		goodInput = true;
		if (boardSize.width <= 0 || boardSize.height <= 0)
		{
			cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << endl;
			goodInput = false;
		}
		if (squareSize <= 10e-6)
		{
			cerr << "Invalid square size " << squareSize << endl;
			goodInput = false;
		}
		if (nrFrames <= 0)
		{
			cerr << "Invalid number of frames " << nrFrames << endl;
			goodInput = false;
		}

		if (input.empty())      // Check for valid input
			inputType = INVALID;
		else
		{
			if (input[0] >= '0' && input[0] <= '9')
			{
				stringstream ss(input);
				ss >> cameraID;
				inputType = CAMERA;
			}
			else
			{
				if (readStringList(input, imageList))
				{
					inputType = IMAGE_LIST;
					nrFrames = (nrFrames < (int)imageList.size()) ? nrFrames : (int)imageList.size();
				}
				else
					inputType = VIDEO_FILE;
			}
			if (inputType == CAMERA)
				inputCapture.open(cameraID);
			if (inputType == VIDEO_FILE)
				inputCapture.open(input);
			if (inputType != IMAGE_LIST && !inputCapture.isOpened())
				inputType = INVALID;
		}
		if (inputType == INVALID)
		{
			cerr << " Inexistent input: " << input;
			goodInput = false;
		}

		flag = 0;
		if (calibFixPrincipalPoint) flag |= CV_CALIB_FIX_PRINCIPAL_POINT;
		if (calibZeroTangentDist)   flag |= CV_CALIB_ZERO_TANGENT_DIST;
		if (aspectRatio)            flag |= CV_CALIB_FIX_ASPECT_RATIO;


		calibrationPattern = NOT_EXISTING;
		if (!patternToUse.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
		if (!patternToUse.compare("CIRCLES_GRID")) calibrationPattern = CIRCLES_GRID;
		if (!patternToUse.compare("ASYMMETRIC_CIRCLES_GRID")) calibrationPattern = ASYMMETRIC_CIRCLES_GRID;
		if (calibrationPattern == NOT_EXISTING)
		{
			cerr << " Inexistent camera calibration mode: " << patternToUse << endl;
			goodInput = false;
		}
		atImageList = 0;

	}
	Mat nextImage()
	{
		Mat result;
		if (inputCapture.isOpened())
		{
			Mat view0;
			inputCapture >> view0;
			view0.copyTo(result);
		}
		else if (atImageList < (int)imageList.size())
			result = imread(imageList[atImageList++], CV_LOAD_IMAGE_COLOR);

		return result;
	}

	static bool readStringList(const string& filename, vector<string>& l)
	{
		l.clear();
		FileStorage fs(filename, FileStorage::READ);
		if (!fs.isOpened())
			return false;
		FileNode n = fs.getFirstTopLevelNode();
		if (n.type() != FileNode::SEQ)
			return false;
		FileNodeIterator it = n.begin(), it_end = n.end();
		for (; it != it_end; ++it)
			l.push_back((string)*it);
		return true;
	}
public:
	Size boardSize;            // The size of the board -> Number of items by width and height
	Pattern calibrationPattern;// One of the Chessboard, circles, or asymmetric circle pattern
	float squareSize;          // The size of a square in your defined unit (point, millimeter,etc).
	int nrFrames;              // The number of frames to use from the input for calibration
	float aspectRatio;         // The aspect ratio
	int delay;                 // In case of a video input
	bool bwritePoints;         //  Write detected feature points
	bool bwriteExtrinsics;     // Write extrinsic parameters
	bool calibZeroTangentDist; // Assume zero tangential distortion
	bool calibFixPrincipalPoint;// Fix the principal point at the center
	bool flipVertical;          // Flip the captured images around the horizontal axis
	string outputFileName;      // The name of the file where to write
	bool showUndistorsed;       // Show undistorted images after calibration
	string input;               // The input ->



	int cameraID;
	vector<string> imageList;
	int atImageList;
	VideoCapture inputCapture;
	InputType inputType;
	bool goodInput;
	int flag;

private:
	string patternToUse;


};

static void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
{
	if (node.empty())
		x = default_value;
	else
		x.read(node);
}

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

calibrationResult runCalibrationAndSave(Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs,
	vector<vector<Point2f> > imagePoints);

static void adaptativeStructuralElement(calibrationResult res){
	cout << res.ok;
};

calibrationResult FilesCalibration::StartFilesCalibration()
{
	struct calibrationResult res;
	res.ok = false;
	help();
	Settings s;
	const std::string inputSettingsFile = "in_VID5.xml";
	FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
	if (!fs.isOpened())
	{
		cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
		res.msg = "Could not open the configuration file :" + inputSettingsFile;
		return res;
	}
	fs["Settings"] >> s;
	fs.release();                                         // close Settings file

	if (!s.goodInput)
	{
		cout << "Invalid input detected. Application stopping. " << endl;

		res.msg = "Invalid input detected. Application stopping. ";
		return res;
	}

	vector<vector<Point2f> > imagePoints;
	Mat cameraMatrix, distCoeffs;
	Size imageSize;
	int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
	clock_t prevTimestamp = 0;
	const Scalar RED(0, 0, 255), GREEN(0, 255, 0);
	const char ESC_KEY = 27;
	for (int i = 0;; ++i)
	{
		Mat view;
		bool blinkOutput = false;

		view = s.nextImage();

		//-----  If no more image, or got enough, then stop calibration and show result -------------
		if (mode == CAPTURING && imagePoints.size() >= (unsigned)s.nrFrames)
		{
			res = runCalibrationAndSave(s, imageSize, cameraMatrix, distCoeffs, imagePoints);
			if (res.ok)
				mode = CALIBRATED;
			else
				mode = DETECTION;
		}
		if (view.empty())          // If no more images then run calibration, save and stop loop.
		{
			if (imagePoints.size() > 0)
				runCalibrationAndSave(s, imageSize, cameraMatrix, distCoeffs, imagePoints);
			break;
		}


		imageSize = view.size();  // Format input image.
		if (s.flipVertical)    flip(view, view, 0);

		vector<Point2f> pointBuf;

		bool found;
		switch (s.calibrationPattern) // Find feature points on the input format
		{
		case Settings::CHESSBOARD:
			found = findChessboardCorners(view, s.boardSize, pointBuf,
				CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
			break;
		case Settings::CIRCLES_GRID:
			found = findCirclesGrid(view, s.boardSize, pointBuf);
			break;
		case Settings::ASYMMETRIC_CIRCLES_GRID:
			found = findCirclesGrid(view, s.boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
			break;
		default:
			found = false;
			break;
		}

		if (found)                // If done with success,
		{
			// improve the found corners' coordinate accuracy for chessboard
			if (s.calibrationPattern == Settings::CHESSBOARD)
			{
				Mat viewGray;
				cvtColor(view, viewGray, COLOR_BGR2GRAY);
				cornerSubPix(viewGray, pointBuf, Size(11, 11),
					Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			}

			if (mode == CAPTURING &&  // For camera only take new samples after delay time
				(!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC))
			{
				imagePoints.push_back(pointBuf);
				prevTimestamp = clock();
				blinkOutput = s.inputCapture.isOpened();
			}

			// Draw the corners.
			drawChessboardCorners(view, s.boardSize, Mat(pointBuf), found);
		}

		//----------------------------- Output Text ------------------------------------------------
		string msg = (mode == CAPTURING) ? "100/100" :
			mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

		if (mode == CAPTURING)
		{
			if (s.showUndistorsed)
				msg = format("%d/%d Undist", (int)imagePoints.size(), s.nrFrames);
			else
				msg = format("%d/%d", (int)imagePoints.size(), s.nrFrames);
		}

		putText(view, msg, textOrigin, 1, 1, mode == CALIBRATED ? GREEN : RED);

		if (blinkOutput)
			bitwise_not(view, view);

		//------------------------- Video capture  output  undistorted ------------------------------
		if (mode == CALIBRATED && s.showUndistorsed)
		{
			Mat temp = view.clone();
			undistort(temp, view, cameraMatrix, distCoeffs);
		}

		//------------------------------ Show image and check for input commands -------------------

		//imshow("Image View", view);
		char key = (char)waitKey(s.inputCapture.isOpened() ? 50 : s.delay);

		if (key == ESC_KEY)
			break;

		if (key == 'u' && mode == CALIBRATED)
			s.showUndistorsed = !s.showUndistorsed;

		if (s.inputCapture.isOpened() && key == 'g')
		{
			mode = CAPTURING;
			imagePoints.clear();
		}
	}

	// -----------------------Show the undistorted image for the image list ------------------------
	if (s.inputType == Settings::IMAGE_LIST && s.showUndistorsed)
	{
		
		int lastImgIndex = s.imageList.size() - 1;
		Mat view, rview, map1, map2, grayView, grayView0;
		initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
			getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
			imageSize, CV_16SC2, map1, map2);

		vector<Point3d> objectPoints;
		vector<Point2d> imagePoints;
		vector<Point3d> objectPointCenter;
		vector<Point2d> imagePointCenter;
		vector<Point3d> lineObjectPoints;
		vector<Point2d> lineImagePoints;

		objectPoints.push_back(Point3d(106 + 26.5, 106 + 26.5, 0));
		objectPoints.push_back(Point3d(106 + 26.5, 106 + 26.5, -26.5));
		objectPointCenter.push_back(Point3d(106 + 26.5, 106 + 26.5, -26.5 / 2));

		objectPoints.push_back(Point3d(-106 - 26.5, 26.5, 0 - 200));
		objectPoints.push_back(Point3d(-106 - 26.5, 26.5, -26.5 - 200));
		objectPointCenter.push_back(Point3d(-106 - 26.5, 26.5, (-26.5 - 200) / 2));
		objectPointCenter.push_back(Point3d(-106 - 26.5, 26.5, (-26.5 - 200) / 2));

		lineObjectPoints.push_back(Point3d(0, 0, 0));
		lineObjectPoints.push_back(Point3d(50, 0, 0));
		lineObjectPoints.push_back(Point3d(0, 50, 0));
		lineObjectPoints.push_back(Point3d(0, 0, 50));

		/*objectPoints.push_back(Point3d(100, 100, 0));
		objectPoints.push_back(Point3d(130, 100, 0));
		objectPoints.push_back(Point3d(50, 50, 0));
		objectPoints.push_back(Point3d(50, 80, 0));*/


		cv::projectPoints(Mat(objectPoints), res.rvecs[lastImgIndex], res.tvecs[lastImgIndex], res.cameraMatrix,
			res.distCoeffs, imagePoints);

		cv::projectPoints(Mat(objectPointCenter), res.rvecs[lastImgIndex], res.tvecs[lastImgIndex], res.cameraMatrix,
			res.distCoeffs, imagePointCenter);

		cv::projectPoints(Mat(lineObjectPoints), res.rvecs[lastImgIndex], res.tvecs[lastImgIndex], res.cameraMatrix,
			res.distCoeffs, lineImagePoints);

		/*pair<double, double> elVector = make_pair(imagePoints[1].x - imagePoints[0].x, imagePoints[1].y - imagePoints[0].y);
		double norm = sqrt(elVector.first*elVector.first + elVector.second*elVector.second);
		double cosinus = elVector.first / norm;
		double sinus = elVector.second / norm;
		int angle = (int)(acos(cosinus) * 180 / PI * (sinus > 0 ? -1 : 1)*(sinus == 0 ? 0 : 1) + 360) % 360;
		*/
		pair<double, double> elVector;
		double norm;
		double cosinus;
		double sinus;
		int angle;


		/*double ch = cos(0.0835);
		double sh = sin(0.0835);
		double ca = cos(0.7896);
		double sa = sin(0.7896);
		double cb = cos(0.05059);
		double sb = sin(0.05059);

		double m00 = ch * ca;
		double m01 = sh*sb - ch*sa*cb;
		double m02 = ch*sa*sb + sh*cb;
		double m10 = sa;
		double m11 = ca*cb;
		double m12 = -ca*sb;
		double m20 = -sh*ca;
		double m21 = sh*sa*cb + ch*sb;
		double m22 = -sh*sa*sb + ch*cb;*/

		/*	m00 = ch * ca;aa
			m01 = sh*sb - ch*sa*cb;
			m02 = ch*sa*sb + sh*cb;
			m10 = sa;
			m11 = ca*cb;
			m12 = -ca*sb;
			m20 = -sh*ca;
			m21 = sh*sa*cb + ch*sb;
			m22 = -sh*sa*sb + ch*cb;*/

		AfxMessageBox(CString("Calibrated!"), MB_OK | MB_ICONEXCLAMATION);
		for (int i = lastImgIndex; i < lastImgIndex + 1; i++) //dla ostatniego obrazka
		{
			view = imread(s.imageList[i], 1);
			if (view.empty())
				continue;
			remap(view, rview, map1, map2, INTER_LINEAR);
			imshow("Image Undistorted View", rview);
			for (int j = 0; j < objectPoints.size(); j = j + 2)
			{
				elVector = make_pair(imagePoints[j + 1].x - imagePoints[j].x, imagePoints[j + 1].y - imagePoints[j].y);
				norm = sqrt(elVector.first*elVector.first + elVector.second*elVector.second);
				cosinus = elVector.first / norm;
				sinus = elVector.second / norm;
				angle = (int)(acos(cosinus) * 180 / PI * (sinus > 0 ? -1 : 1)*(sinus == 0 ? 0 : 1) + 360) % 360;
				ellipse(view, imagePointCenter[j], Size(norm *0.5, norm*0.5*0.5), angle, 0, 360, Scalar(0, 0, 0), -1, 8, 0);
				//ellipse(view, Point(0, 0), Size(10, 30), 120, 0, 360, Scalar(0, 0, 0), -1, 8, 0);
			}

			cvtColor(view, grayView, CV_BGR2GRAY);

			ocl::oclMat oclSrc = ocl::oclMat(grayView);
			ocl::oclMat oclDst;

			arrowedLine(view, lineImagePoints[0], lineImagePoints[1], Scalar(0, 0, 255), 5, 8, 0);//x red
			arrowedLine(view, lineImagePoints[0], lineImagePoints[2], Scalar(0, 255, 0), 5, 8, 0);//y green
			arrowedLine(view, lineImagePoints[0], lineImagePoints[3], Scalar(255, 0, 0), 5, 8, 0);//z blue

			freopen("CONOUT$", "w", stdout);
			cout << "aaaaaaaaaaaaaaaa";

			cv::ocl::DevicesInfo devInfo;
			int res = cv::ocl::getOpenCLDevices(devInfo);
			if (res == 0)
			{
				std::cerr << "There is no OPENCL Here !" << std::endl;
			}
			else
			{
				for (unsigned int i = 0; i < devInfo.size(); ++i)
				{
					std::cout << "Device : " << devInfo[i]->deviceName << " is present" << std::endl;
				}
			}

			cv::ocl::setDevice(devInfo[0]);        // select device to use
			std::cout << CV_VERSION_EPOCH << "." << CV_VERSION_MAJOR << "." << CV_VERSION_MINOR << std::endl;

			const char *KernelSource = "\n" \
				"__kernel void negaposi_C1_D0(               \n" \
				"   __global uchar* input,                   \n" \
				"   __global uchar* output)                  \n" \
				"{                                           \n" \
				"   int i = get_global_id(0);                \n" \
				"   output[i] = 255 - input[i];              \n" \
				"}\n";

			cv::Mat mat_src = cv::imread("lena.jpg", cv::IMREAD_GRAYSCALE);
			cv::Mat mat_dst;
			if (mat_src.empty())
			{
				std::cerr << "Failed to open image file." << std::endl;
			}
			unsigned int channels = mat_src.channels();
			unsigned int depth = mat_src.depth();

			cv::ocl::oclMat ocl_src(mat_src);
			cv::ocl::oclMat ocl_dst(mat_src.size(), mat_src.type());

			cv::ocl::ProgramSource program("negaposi", KernelSource);
			std::size_t globalThreads[3] = { ocl_src.rows * ocl_src.step, 1, 1 };
			std::vector<std::pair<size_t, const void *> > args;
			args.push_back(std::make_pair(sizeof(cl_mem), (void *)&ocl_src.data));
			args.push_back(std::make_pair(sizeof(cl_mem), (void *)&ocl_dst.data));
			cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
				program, "negaposi", globalThreads, NULL, args, channels, depth, NULL);
			ocl_dst.download(mat_dst);

			cv::namedWindow("mat_src");
			cv::namedWindow("mat_dst");
			cv::imshow("mat_src", mat_src);
			cv::imshow("mat_dst", mat_dst);
			cv::waitKey(0);
			cv::destroyAllWindows();

			const clock_t begin_time2 = clock();
			for (int i = 0; i < 1; i++)
			{
				threshold(grayView, grayView0, 200, 255, THRESH_BINARY);
			}
			float ocv = float(clock() - begin_time2) / CLOCKS_PER_SEC;

			const clock_t begin_time = clock();
			for (int i = 0; i < 1; i++)
			{
				cv::ocl::threshold(oclSrc, oclDst, 200, 255, THRESH_BINARY);
			}
			float ocl = float(clock() - begin_time) / CLOCKS_PER_SEC;
			/*static const ocl::DeviceInfo devicesInfo = ocl::DeviceInfo();*/
			ocl::PlatformsInfo platforms;
			int platformCount = ocl::getOpenCLPlatforms(platforms);

			ocl::DevicesInfo devices;
			int devicesCount = ocl::getOpenCLDevices(devices);
			ocl::oclMat afterErodeOcl;
			Mat afterErodeOcv;
			Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(31, 51));
			ocl::oclMat kernelOCL = ocl::oclMat(kernel);
			//auto B = new int[10][20];
			//for (int i = 0; i<kernel.rows; i++){ 
			//	uchar* rowi = kernel.ptr/*<uchar>*/(i);
			//	for (int j = 0; j<kernel.cols; j++){
			//		rowi[j] = rowi[j] * 255;
			//	}
			//}
			try
			{
				const clock_t begin_time1 = clock();
				ocl::erode(oclDst, afterErodeOcl, kernelOCL);
				float oclerode = float(clock() - begin_time1) / CLOCKS_PER_SEC;

				const clock_t begin_time2 = clock();
				cv::erode(grayView0, afterErodeOcv, kernel);
				float ocverode = float(clock() - begin_time2) / CLOCKS_PER_SEC;

				//ocl::morphologyEx(oclDst, afterErodeOcl, 1, kernel);
				//2058 2456
				Mat dstFromOcl = Mat(oclDst);
				Mat dsfAfterErodeOcl = Mat(afterErodeOcl);

				
				const char* winName = "threshold";
				namedWindow(winName, 0);
				cvResizeWindow(winName, 800, 600);
				namedWindow("erode", 0);
				cvResizeWindow("erode", 800, 600);
				imshow(winName, afterErodeOcv);
				
				imshow("erode", dsfAfterErodeOcl);
			}
			catch (cv::Exception & e)
			{
				cout << "oo" << endl;
			}
			char c = (char)waitKey();
			if (c == ESC_KEY || c == 'q' || c == 'Q')
				break;
		}

	}

	return res;
};

static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); ++i)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
			distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
	Settings::Pattern patternType = Settings::CHESSBOARD)
{
	corners.clear();

	switch (patternType)
	{
	case Settings::CHESSBOARD:
	case Settings::CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; ++i)
			for (int j = 0; j < boardSize.width; ++j)
				corners.push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
		break;

	case Settings::ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float((2 * j + i % 2)*squareSize), float(i*squareSize), 0));
		break;
	default:
		break;
	}
}

static bool runCalibration(Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs, double& totalAvgErr)
{

	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (s.flag & CV_CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = 1.0;

	distCoeffs = Mat::zeros(8, 1, CV_64F);

	vector<vector<Point3f> > objectPoints(1);
	calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	//Find intrinsic and extrinsic camera parameters
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
		distCoeffs, rvecs, tvecs, s.flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}

// Print camera parameters to the output file
static void saveCameraParams(Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
	double totalAvgErr)
{
	FileStorage fs(s.outputFileName, FileStorage::WRITE);

	time_t tm;
	time(&tm);
	struct tm *t2 = localtime(&tm);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_Time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nrOfFrames" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "image_Width" << imageSize.width;
	fs << "image_Height" << imageSize.height;
	fs << "board_Width" << s.boardSize.width;
	fs << "board_Height" << s.boardSize.height;
	fs << "square_Size" << s.squareSize;

	if (s.flag & CV_CALIB_FIX_ASPECT_RATIO)
		fs << "FixAspectRatio" << s.aspectRatio;

	if (s.flag)
	{
		sprintf(buf, "flags: %s%s%s%s",
			s.flag & CV_CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "",
			s.flag & CV_CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "",
			s.flag & CV_CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "",
			s.flag & CV_CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "");
		cvWriteComment(*fs, buf, 0);

	}

	fs << "flagValue" << s.flag;

	fs << "Camera_Matrix" << cameraMatrix;
	fs << "Distortion_Coefficients" << distCoeffs;

	fs << "Avg_Reprojection_Error" << totalAvgErr;
	if (!reprojErrs.empty())
		fs << "Per_View_Reprojection_Errors" << Mat(reprojErrs);

	if (!rvecs.empty() && !tvecs.empty())
	{

		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
		for (int i = 0; i < (int)rvecs.size(); i++)
		{
			Mat r = bigmat(Range(i, i + 1), Range(0, 3));
			Mat t = bigmat(Range(i, i + 1), Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
			//*.t() is MatExpr (not Mat) so we can use assignment operator
			r = rvecs[i].t();
			t = tvecs[i].t();
		}
		cvWriteComment(*fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0);
		fs << "Extrinsic_Parameters" << bigmat;
		cvWriteComment(*fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0);
		fs << "dsdsdsds" << (int)rvecs.size();
	}

	if (!imagePoints.empty())
	{
		Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (int i = 0; i < (int)imagePoints.size(); i++)
		{
			Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "Image_points" << imagePtMat;
	}
}

calibrationResult runCalibrationAndSave(Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs, vector<vector<Point2f> > imagePoints)
{
	calibrationResult result;
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
		reprojErrs, totalAvgErr);
	cout << (ok ? "Calibration succeeded" : "Calibration failed")
		<< ". avg re projection error = " << totalAvgErr;

	if (ok)
		saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs,
		imagePoints, totalAvgErr);
	result.ok = ok;
	result.rvecs = rvecs;
	result.tvecs = tvecs;
	result.cameraMatrix = cameraMatrix;
	result.distCoeffs = distCoeffs;
	return result;
}
