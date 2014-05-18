#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>

#include "RGBimg.h"
#include<vector>
#include<iostream>
#include "kernel.h"


int main()
{
	int * threshold = new int;
	*threshold = 0;
	cv::Mat img;
	cv::Mat gs;
	cv::Mat testImg;
	cv::VideoCapture cam(0);
	RGBimg frame;	
	cv::namedWindow("ORG");
	cv::namedWindow("Edges");
	cv::createTrackbar("Threshold", "Edges", threshold, 100);

	testImg = cv::imread("sample.png", CV_LOAD_IMAGE_COLOR);
	testImg = detectEdges(testImg, 50);
	cv::namedWindow("test");
	cv::imshow("test",testImg);
	
	while(1)
	{
		cam >> img;
		cv::imshow("ORG",img);
		gs = detectEdges(img, cv::getTrackbarPos("Threshold", "Edges"));
		cv::imshow("Edges",gs);
		cv::waitKey(1);
	}

	cv::waitKey(10000);


}