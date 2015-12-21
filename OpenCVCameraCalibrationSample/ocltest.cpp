#include "stdafx.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include <boost/compute.hpp>
//#include <boost/compute/interop/opencv/core.hpp>
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

	/*const char *KernelSource = "\n" \
		"__kernel void erode(																											\n" \
		"	read_only image2d_t image,																									\n" \
		"	int sizeOfElement, write_only image2d_t imageOut, int imageWidth, int imageHeight) {										\n" \
		"																																\n" \
		"	const int2 coords = { get_global_id(0), get_global_id(1) };																	\n" \
		"	if (coords.x >= imageWidth || coords.y >= imageHeight){																		\n" \
		"		return;																													\n" \
		"	}																															\n" \
		"	const sampler_t sampler =																									\n" \
		"		CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;													\n" \
		"	uint4 ans = { 255, 255, 255, 255 };																							\n" \
		"	for (int i = 0; i < sizeOfElement; ++i) {																					\n" \
		"		const int2 elementCoords = element[i];																					\n" \
		"		const int2 imageCoords = coords;																						\n" \
		"		const uint4 imagePixel = read_imageui(image, sampler, imageCoords);														\n" \
		"		if (imagePixel.x < 255) {																								\n" \
		"			ans.x = 0;																											\n" \
		"			ans.y = 0;																											\n" \
		"			ans.z = 0;																											\n" \
		"			ans.w = 0;																											\n" \
		"			break;																												\n" \
		"		}																														\n" \
		"	}																															\n" \
		"	write_imageui(imageOut, coords, ans);																						\n" \
		"}																																\n";*/

	/*const char *KernelSource = "\n" \
		"__kernel void erode(				                \n" \
			"   __global uchar* input,						\n" \
			"   __global uchar* output)						\n" \
			"{												\n" \
			"   int i = get_global_id(0);					\n" \
			"   output[i] = input[i] >= 100 ? 0 : 255;      \n" \
			"}\n";*/
	
		
	//"__kernel void negaposi_C1_D0(               \n" \
			//"   __global uchar* input,                   \n" \
			//"   __global uchar* output)                  \n" \
			//"{                                           \n" \
			//"   int i = get_global_id(0);                \n" \
			//"   output[i] = input[i] >= 100 ? 0 : 255;              \n" \
			//"}\n";
	
	/*const char sourceSmart[] = BOOST_COMPUTE_STRINGIZE_SOURCE(__kernel void erode(
		read_only image2d_t image, __global read_only int2 *element,
		int sizeOfElement, write_only image2d_t imageOut, int2 elementDim, int imageWidth, int imageHeight) {

		const int2 coords = { get_global_id(0), get_global_id(1) };
		if (coords.x >= imageWidth || coords.y >= imageHeight){
			return;
		}
		const sampler_t sampler =
			CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
		uint4 ans = { 255, 255, 255, 255 };
		for (int i = 0; i < sizeOfElement; ++i) {
			const int2 elementCoords = element[i];
			const int2 imageCoords = coords + elementCoords - (elementDim >> 1);
			const uint4 imagePixel = read_imageui(image, sampler, imageCoords);
			if (imagePixel.x < 255) {
				ans.x = 0;
				ans.y = 0;
				ans.z = 0;
				ans.w = 0;
				break;
			}
		}
		write_imageui(imageOut, coords, ans);
	});*/
	//cl::Image2D aa = cl::Image2D(cv::ocl::Context::getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_UNORM_INT8));
	
	//cv::Mat mat_src = cv::imread("lena.jpg", cv::IMREAD_GRAYSCALE);
	//cv::Mat mat_dst;
	//if (mat_src.empty())
	//{
	//	std::cerr << "Failed to open image file." << std::endl;
	//}
	//unsigned int channels = mat_src.channels();
	//unsigned int depth = mat_src.depth();

	//cv::ocl::oclMat ocl_src(mat_src);
	//cv::ocl::oclMat ocl_dst(mat_src.size(), mat_src.type());
	//cv::Mat notSrc, notDst;
	//notSrc = cv::Mat(ocl_src);
	//cv::ocl::ProgramSource program("erode", KernelSource);
	//std::size_t globalThreads[3] = { ocl_src.rows * ocl_src.step, 1, 1 };
	//std::size_t localThreads[3] = { ocl_src.rows * ocl_src.step, 1, 1 };
	//std::vector<std::pair<size_t, const void *> > args;
	////args.push_back(std::make_pair(sizeof(cl_mem), (void *)&ocl_src.data));
	//
	//cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(50, 50));
	//cv::ocl::oclMat elementOcl = cv::ocl::oclMat(element);
	//args.push_back(std::make_pair(sizeof(cl_mem), (void *)&ocl_src.data));
	////args.push_back(std::make_pair(sizeof(cl_mem), (void *)100));
	//args.push_back(std::make_pair(sizeof(cl_mem), (void *)&ocl_dst.data));
	//args.push_back(std::make_pair(sizeof(cl_mem), (void *)1000));
	//args.push_back(std::make_pair(sizeof(cl_mem), (void *)1000));
	/*read_only image2d_t image, __global read_only int2 *element, int sizeOfElement, write_only image2d_t imageOut, int2 elementDim, int imageWidth, int imageHeight*/
	//cl_mem imageSource = clCreateImage2D(, );

//		"	read_only image2d_t image,																									\n" \
//		"	int sizeOfElement, write_only image2d_t imageOut, int imageWidth, int imageHeight) {										\n" \
	
	/*float c1 = clock();
	
		cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
			program, "erode", globalThreads, NULL, args, channels, depth, NULL);
	
	ocl_dst.download(mat_dst);
	float t1 = (clock() - c1) / CLOCKS_PER_SEC;
	ocl_dst.download(mat_dst);
	float c2 = clock();
	for (int i = 0; i < 200000; i++)
		cv::threshold(notSrc, notDst, 100, 255, 1);
	float t2 = (clock() - c2) / CLOCKS_PER_SEC;*/


	const char *KernelSource = "\n" \
		"__kernel void erode_C1_D0(               \n" \
		"   __global uchar* input,                   \n" \
		"   __global uchar* image,                   \n" \
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

	cv::ocl::ProgramSource program("erode", KernelSource);
	std::size_t globalThreads[3] = { ocl_src.rows * ocl_src.step, 1, 1 };
	std::vector<std::pair<size_t, const void *> > args;
	args.push_back(std::make_pair(sizeof(cl_mem), (void *)&ocl_src.data));
	args.push_back(std::make_pair(sizeof(cl_mem), (void *)&ocl_src.data));
	args.push_back(std::make_pair(sizeof(cl_mem), (void *)&ocl_dst.data));
	cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
		program, "erode", globalThreads, NULL, args, channels, depth, NULL);
	
	ocl_dst.download(mat_dst);

	cv::namedWindow("mat_src");
	cv::namedWindow("ocl");
	cv::imshow("mat_src", mat_src);
	cv::imshow("ocl", mat_dst);
//	cv::imshow("ocv", notDst);
	cv::waitKey(0);
	cv::destroyAllWindows();
};

//boost::compute::image2d convertImage(const cv::Mat &cvimage, cl_mem_flags flags,
//	boost::compute::command_queue &queue, const boost::compute::context &ctx) {
//
//	const boost::compute::image_format format(CL_R, CL_UNSIGNED_INT8);
//	const auto formats = boost::compute::image2d::get_supported_formats(ctx);
//	/*if (std::none_of(begin(formats), end(formats),[=](auto supFor) { return supFor == format; })) {
//		assert(false);
//	}*/
//	using namespace boost::compute;
//
//	using std::tuple;
//	using std::make_tuple;
//	
//	using cv::Mat;
//	image2d image(ctx, cvimage.cols, cvimage.rows, format, flags);
//	opencv_copy_mat_to_image(cvimage, image, queue);
//	return image;
//};