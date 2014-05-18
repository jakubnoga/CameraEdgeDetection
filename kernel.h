#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2\core\core.hpp>
#include "RGBimg.h"
#include <stdio.h>

cv::Mat rgb2gs(cv::Mat rgb);
__global__ void rgb2gsKernel(uchar * r, uchar * g, uchar * b, uchar * gs, int rows, int cols);
cv::Mat gaussOnGs(cv::Mat gs);
__global__ void gaussOnGsKernel(unsigned char * input, unsigned char * output, int rows, int cols);
__global__ void applyFilterKernel(unsigned char * input, unsigned char * output, float * filter, int rows, int cols, int fsize);
cv::Mat applyFilter(cv::Mat chn, float * filter, int fsize);
cv::Mat detectEdges(cv::Mat img, int threshold);
cv::Mat magnitude(cv::Mat img1, cv::Mat img2, int threshold);
__global__ void magnitudeKernel(unsigned char * input1, unsigned char * input2, unsigned char * output, int rows, int cols, int threshold);