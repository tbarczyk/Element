#include "stdafx.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/ocl/ocl.hpp"
#pragma comment (lib,"OpenCL.lib")
#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

void testOCL() {
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
}