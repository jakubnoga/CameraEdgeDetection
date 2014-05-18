#include "RGBimg.h"


RGBimg::RGBimg(void)
{
	R = nullptr;
	G = nullptr;
	B = nullptr;

	rows = 0;
	cols = 0;

}


RGBimg::~RGBimg(void)
{
}


void RGBimg::mat2rgb(cv::Mat img)
{
	if (B)
	delete B;
	if (G)
	delete G;
	if (R)
	delete R;

	rows = img.rows;
	cols = img.cols;

	std::vector<cv::Mat> chns;

	B = new unsigned char[img.cols*img.rows];
	G = new unsigned char[img.cols*img.rows];
	R = new unsigned char[img.cols*img.rows];

	cv::split(img,chns);
	for(int i = 0; i< cols*rows; i++)
	{
		B[i] = chns.at(0).data[i];
		G[i] = chns.at(1).data[i];
		R[i] = chns.at(2).data[i];
	}


}

cv::Mat RGBimg::rgb2gs()
{
	unsigned char * gs;
	float c;
	gs = new unsigned char[cols*rows];

	for(int i = 0; i<cols*rows;i++)
	{
		c = (float)R[i]*0.21+(float)G[i]*0.72+(float)B[i]*0.07;
		gs[i] = (unsigned char)c;
	}
	
	cv::Mat result(cv::Size(cols,rows),CV_8UC1,gs);
	return result;
}