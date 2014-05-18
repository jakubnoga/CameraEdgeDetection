#pragma once
#include<opencv2\core\core.hpp>
#include<vector>
#include<iostream>

class RGBimg
{
public:
	RGBimg(void);
	~RGBimg(void);

	int rows;
	int cols;

	unsigned char * R;
	unsigned char * G;
	unsigned char * B;

	void mat2rgb(cv::Mat img);

	cv::Mat rgb2gs();
};

