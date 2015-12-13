#include "stdafx.h"
#include "Element.h"
#define PI 3.1415


using namespace std;
using namespace cv;

Element::Element()
{
	/*angle = 0;
	p1 = *(new Point2d(0,0));
	p2 = *(new Point2d(0, 0));*/
}

void Element::ComputeElement(calibrationResult res)
{
	// these are real points in milimeters
	vector<Point3d> objectPoints;
	vector<Point2d> imagePoints;
	objectPoints.push_back(Point3d(0, 0, 0));
	objectPoints.push_back(Point3d(0, 0, 60));
	/*objectPoints.push_back(Point3d(60, 0, 0));
	objectPoints.push_back(Point3d(0, 60, 0));
	objectPoints.push_back(Point3d(0, 120, 60));
	objectPoints.push_back(Point3d(120, 0, 60));*/
	//objectPoints.push_back(Point3d(10000, 15000, 0));
	//objectPoints.push_back(Point3d(0, 0, 0));
	//objectPoints.push_back(Point3d(30, 30, 0));
	//objectPoints.push_back(Point3d(0, 0, 1000));
	//objectPoints.push_back(Point3d(500, 500, 0));
	cv::projectPoints(Mat(objectPoints), res.rvecs[8], res.tvecs[8], res.cameraMatrix,
		res.distCoeffs, imagePoints);
	
	pair<double, double> elVector = make_pair(imagePoints[1].x - imagePoints[0].x, imagePoints[1].y - imagePoints[0].y);
	double norm = sqrt(elVector.first*elVector.first + elVector.second*elVector.second);
	double cos = elVector.first / norm;
	double sin = elVector.second / norm;
	//angle = (int)(acos(cos) * 180 / PI * (sin > 0 ? -1 : 1)*(sin == 0 ? 0 : 1) + 360) % 360;
	//p1 = imagePoints[0];
	//p2 = imagePoints[1];
	//
	////ellipse(view, Point(50, 50), Size(10, 30), 30, 0, 360, Scalar(0, 0, 0), -1, 8, 0);
}

void Element::GenerateElements(int width, int height)
{

}

Element::~Element()
{
}
