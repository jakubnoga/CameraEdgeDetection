
#include "kernel.h"
#include <cmath>
#include <vector>

__global__ void rgb2gsKernel(unsigned char * r, unsigned char * g, unsigned char * b, unsigned char * gs, int rows, int cols)
{
	int blockId = gridDim.x * blockIdx.y + blockIdx.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	float c;

	if(threadId < rows*cols)
	{
		c = (float)(r[threadId])*0.21+(float)(g[threadId])*0.72+(float)(b[threadId])*0.07;
		gs[threadId] = (unsigned char)c;
	}
	
}


cv::Mat rgb2gs(cv::Mat rgb)
{
	RGBimg frame;
	frame.mat2rgb(rgb);

	int x = frame.cols, y = frame.rows;
	int size = frame.cols*frame.rows;

	dim3 blockSize;
	blockSize.x = 32;
	blockSize.y = 32;

	dim3 gridSize;
	gridSize.x = frame.cols/blockSize.x + 1;
	gridSize.y = frame.rows/blockSize.y + 1;

	unsigned char *d_r, *d_g, *d_b, *d_gs;
	unsigned char *h_gs;

	h_gs = new unsigned char[size];

	cudaMalloc((void**)&d_r, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_g, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_b, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_gs, size*sizeof(unsigned char));

	
	cudaMemcpy(d_r, frame.R,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_g, frame.G,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, frame.B,size*sizeof(unsigned char),cudaMemcpyHostToDevice);

	rgb2gsKernel<<< gridSize, blockSize >>>(d_r, d_g, d_b, d_gs, x, y);

	cudaMemcpy(h_gs,d_gs,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);

	cv::Mat result(cv::Size(frame.cols,frame.rows),CV_8UC1,h_gs);

	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);
	cudaFree(d_gs);

	return result;

}

__global__ void gaussOnGsKernel(unsigned char * input, unsigned char * output, int rows, int cols)
{
	int blockId = gridDim.x * blockIdx.y + blockIdx.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	float c = 0;

	if (threadId < rows*cols && threadId > cols - 1 )
	{
		c = 4*input[threadId]+2*(input[threadId-1]+input[threadId+1]+input[threadId-rows]+input[threadId+rows]);
		c += input[threadId-rows+1] + input[threadId-rows-1] + input[threadId+rows+1] + input[threadId+rows-1];
		c /= 16;
		output[threadId] = (unsigned char)c;
	}

	
}


cv::Mat gaussOnGs(cv::Mat gs)
{
	int x = gs.cols, y = gs.rows;
	int size = gs.cols*gs.rows;

	dim3 blockSize;
	blockSize.x = 32;
	blockSize.y = 32;

	dim3 gridSize;
	gridSize.x = gs.cols/blockSize.x + 1;
	gridSize.y = gs.rows/blockSize.y + 1;

	unsigned char *d_input, *d_output;
	unsigned char *h_output;

	h_output = new unsigned char[size];

	cudaMalloc((void**)&d_input, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_output, size*sizeof(unsigned char));


	
	cudaMemcpy(d_input, gs.data, size*sizeof(unsigned char),cudaMemcpyHostToDevice);


	gaussOnGsKernel<<< gridSize, blockSize >>>(d_input, d_output, x, y);

	cudaMemcpy(h_output,d_output,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);

	cv::Mat result(cv::Size(gs.cols,gs.rows),CV_8UC1,h_output);

	cudaFree(d_input);
	cudaFree(d_output);

	return result;

}

cv::Mat applyFilter(cv::Mat chn, float * filter, int fsize)
{
	int size = chn.cols*chn.rows;

	dim3 blockSize;
	blockSize.x = 32;
	blockSize.y = 32;

	dim3 gridSize;
	gridSize.x = chn.cols/blockSize.x + 1;
	gridSize.y = chn.rows/blockSize.y + 1;

	unsigned char *d_input, *d_output;
	float *d_filter;
	unsigned char *h_output;

	h_output = new unsigned char[size];

	cudaMalloc((void**)&d_input, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_output, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_filter, fsize*fsize*sizeof(float));


	
	cudaMemcpy(d_input, chn.data, size*sizeof(unsigned char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, filter, fsize*fsize*sizeof(float),cudaMemcpyHostToDevice);

	applyFilterKernel<<< gridSize, blockSize >>>(d_input, d_output, d_filter, chn.rows, chn.cols, fsize);

	cudaMemcpy(h_output, d_output, size*sizeof(unsigned char),cudaMemcpyDeviceToHost);

	cv::Mat result(cv::Size(chn.cols, chn.rows),CV_8UC1,h_output);

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_filter);

	return result;
}

__global__ void applyFilterKernel(unsigned char * input, unsigned char * output, float * filter, int rows, int cols, int fsize)
{
	int y = blockIdx.x*blockDim.x + threadIdx.x;
	int x = blockIdx.y*blockDim.y + threadIdx.y;

	float c = 0;
	int flength = fsize*fsize;
	
	if ((y > fsize/2-1 && x < cols - fsize/2 ) || (y < rows-fsize/2 && x > fsize/2 -1))
	{
		for(int i = 0; i < flength; i++)
		{
			int yoff = i/fsize - fsize/2;
			int xoff = i % fsize - fsize/2;

			c += (float)input[(x+xoff)*cols + y+yoff]*filter[i];
		}
	}
	if (c > 255) c = 255;
	if (c < 0) c = 0;

	output[x*cols + y] = (unsigned char)c;
}

__global__ void magnitudeKernel(unsigned char * input1, unsigned char * input2, unsigned char * output, int rows, int cols, int threshold)
{
	int blockId = gridDim.x * blockIdx.y + blockIdx.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	double c=0;

	if( threadId < rows*cols)
		c = sqrt((pow((double)input1[threadId],2)+pow((double)input2[threadId],2)));
	if (c > 255) c = 255;
	if (c < threshold) c = 0;

	output[threadId] = (unsigned char)c;

}


cv::Mat magnitude(cv::Mat img1, cv::Mat img2, int threshold)
{
	int size = img1.cols*img1.rows;

	dim3 blockSize;
	blockSize.x = 32;
	blockSize.y = 32;

	dim3 gridSize;
	gridSize.x = img1.cols/blockSize.x + 1;
	gridSize.y = img1.rows/blockSize.y + 1;

	unsigned char *d_input1, *d_input2, *d_output;
	unsigned char *h_output;

	h_output = new unsigned char[size];

	cudaMalloc((void**)&d_input1, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_input2, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_output, size*sizeof(unsigned char));
	
	cudaMemcpy(d_input1, img1.data, size*sizeof(unsigned char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_input2, img2.data, size*sizeof(unsigned char),cudaMemcpyHostToDevice);


	magnitudeKernel<<< gridSize, blockSize >>>(d_input1, d_input2, d_output, img1.rows, img1.cols, threshold);

	cudaMemcpy(h_output, d_output, size*sizeof(unsigned char),cudaMemcpyDeviceToHost);

	cv::Mat result(cv::Size(img1.cols, img1.rows),CV_8UC1,h_output);


	cudaFree(d_input1);
	cudaFree(d_input2);
	cudaFree(d_output);

	return result;
}


cv::Mat detectEdges(cv::Mat img, int threshold)
{
	cv::Mat temp, temp1, temp2;
	cv::Mat Gx, Gy;
	temp = rgb2gs(img);





	float sobel1[9] = {-1, 0, 1,
						-2, 0, 2,
						-1, 0, 1};

	float sobel2[9] = {-1, -2, -1,
						0, 0, 0,
						1, 2, 1};

	float sobel3[9] = {0, 1, 2,
						-1, 0, 1,
						-2, -1, };

	float sobel4[9] = {2, 1, 0,
						1, 0, -1,
						0, -1, -2};


	float gauss5[25] = {1.0, 4.0, 7.0, 4.0, 1.0, 
						4.0, 16.0, 26.0, 16.0, 4.0,
						7.0, 26.0, 41.0, 26.0, 7.0,
						4.0, 16.0, 26.0, 16.0, 4.0,
						1.0, 4.0, 7.0, 4.0, 1.0};

	float gauss3[9] = {1.0, 2.0, 1.0,
						2.0, 4.0, 2.0,
						1.0, 2.0, 1.0};

	float sharpen[9] = {-1, -1, -1,
						-1, 8, -1,
						-1, -1, -1};

	float enhance[9] = {0, -1, 0,
						-1, 4, -1,
						0, -1, 0};

	for (int i=0; i<9;i++)
		gauss3[i] /= 16;

	for (int i=0; i<25;i++)
		gauss5[i] /= 273;


	temp = applyFilter(temp, gauss5, 5);

	Gx = applyFilter(temp, sobel1, 3);
	Gy = applyFilter(temp, sobel3, 3);
	temp1 = magnitude(Gx,Gy, threshold);

	delete Gx.data;
	delete Gy.data;

	Gx = applyFilter(temp, sobel2, 3);
	Gy = applyFilter(temp, sobel4, 3);
	temp2 = magnitude(Gx,Gy, threshold);

	delete Gx.data;
	delete Gy.data;

	temp2 = magnitude(temp1,temp2, threshold);

	temp2 = applyFilter(temp2, sharpen, 3);

	return temp2;
}


/* 3 kana³y */

//cv::Mat detectEdges(cv::Mat img)
//{
//	cv::Mat *tempB, *tempG, *tempR, temp;
//	cv::Mat GxB, GyB,GxG, GyG,GxR, GyR;
//	//temp = rgb2gs(img);
//
//	std::vector<cv::Mat> chns;
//
//	cv::split(img,chns);
//	
//	tempB = &(chns[0]);
//	tempG = &(chns[1]);
//	tempR = &(chns[2]);
//
//
//	float xsobel[9] = {-1, 0, 1,
//						-2, 0, 2,
//						-1, 0, 1};
//
//	float ysobel[9] = {-1, -2, -1,
//						0, 0, 0,
//						1, 2, 1};
//
//	float gauss5[25] = {1.0, 4.0, 7.0, 4.0, 1.0, 
//						4.0, 16.0, 26.0, 16.0, 4.0,
//						7.0, 26.0, 41.0, 26.0, 7.0,
//						4.0, 16.0, 26.0, 16.0, 4.0,
//						1.0, 4.0, 7.0, 4.0, 1.0};
//
//	float gauss3[9] = {1.0, 2.0, 1.0,
//						2.0, 4.0, 2.0,
//						1.0, 2.0, 1.0};
//
//	for (int i=0; i<9;i++)
//		gauss3[i] /= 16;
//
//	for (int i=0; i<25;i++)
//		gauss5[i] /= 273;
//
//
//	*tempB = applyFilter(*tempB, gauss5, 5);
//	*tempG = applyFilter(*tempG, gauss5, 5);
//	*tempR = applyFilter(*tempR, gauss5, 5);
//
//	GxB = applyFilter(*tempB, xsobel, 3);
//	GyB = applyFilter(*tempB, ysobel, 3);
//	GxG = applyFilter(*tempG, xsobel, 3);
//	GyG = applyFilter(*tempG, ysobel, 3);
//	GxR = applyFilter(*tempR, xsobel, 3);
//	GyR = applyFilter(*tempR, ysobel, 3);
//
//	*tempB = magnitude(GxB,GyB);
//	*tempG = magnitude(GxG,GyG);
//	*tempR = magnitude(GxR,GyR);
//
//	cv::merge(chns,temp);
//
//	return temp;
//}