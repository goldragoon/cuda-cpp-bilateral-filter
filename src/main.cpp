#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cxxopts.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void bilateralFilter(const Mat & input, Mat & output, int r,double sI, double sS);

int main(int argc, char** argv) {

	cxxopts::Options options("Preprocessing", "paganinist@gmail.com");
	options.add_options()
	    ("i,input", "", cxxopts::value<std::string>())
	    ("o,output", "", cxxopts::value<std::string>());
	
	auto result = options.parse(argc, argv);

	if(result.count("help")) {
		std::cout << options.help() << std::endl;
		return 0;
	}

	std::string input_fname, output_fname;
	if(result.count("input")) {
		input_fname = result["input"].as<std::string>();
	} else {
		return 1;
	}

	if(result.count("output")) {
		output_fname = result["output"].as<std::string>();
	} else {
		return 1;
	}

	clock_t s, e, inner_s, inner_e;
		
	Mat input_img = imread(input_fname.c_str(), IMREAD_GRAYSCALE);	

	cout << "[cv::bilateralFilter]" << endl; 
	// OpenCV CPU bilateral filter
	Mat output_opencv_cpu;
	s = clock();
	cv::bilateralFilter(input_img, output_opencv_cpu, 9, 75, 75);
	e = clock();
	cout << "Total CPU Time : " << (e - s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << endl;

	cout << "[cv::cuda::bilateralFilter]" << endl; 
	// OpenCV GPU bilateral filter
	s = clock();
	cv::cuda::GpuMat input_gpuimg;
	input_gpuimg.upload(input_img);
	// Above two lines can be replaced with 'cv::cuda::GpuMat(input_img2);'
	
        cv::cuda::GpuMat output_gpuimg;
	inner_s = clock();	
	cv::cuda::bilateralFilter(input_gpuimg, output_gpuimg, 9, 75.0, 75.0);	
	inner_e = clock();
	cv::Mat output_opencv_gpu = cv::Mat(output_gpuimg);
	e = clock();
	cout << "CPU Time without memcopy: " << (inner_e - inner_s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << endl;
	cout << "Total CPU Time : " << (e - s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << endl;
	
	// Own bilateral filter (input,output,filter_half_size,sigmaI,sigmaS)
	cout << "[My bilateralFilter]" << endl;
	Mat output_my(input_img.rows, input_img.cols, CV_8UC1);
	//cv::Mat output_my;
	s = clock();
	bilateralFilter(input_img, output_my, 4, 75.0, 75.0);
	e = clock();
	cout << "Total CPU Time : " << (e - s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << endl;
	
	imwrite((std::string("opencv_cpu_") + output_fname).c_str(), output_opencv_cpu);
	imwrite((std::string("opencv_gpu_") + output_fname).c_str(), output_opencv_gpu);
	imwrite((std::string("my_") + output_fname).c_str(), output_my);	
}
