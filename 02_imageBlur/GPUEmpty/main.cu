/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Exercise: blur an image by averaging each pixel's
 *                  neighborhood on both the CPU and GPU, and compare the
 *                  execution time of each. Everything except the GPU
 *                  kernel is already written - fill in the //MK: TODO in
 *                  gpuCode() below.
 *
 *        Version:  1.0
 *        Created:  08/16/2026 02:10:00 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Yoon, Myung Kuk, myungkuk.yoon@ewha.ac.kr
 *   Organization:  EWHA Womans Unversity
 *
 * =====================================================================================
 */

#include <stdio.h>
#include "../../common/ppm.h"
#include "../../common/clockMeasure.h"
#include "../../common/cudaCommon.h"

#define BLUR_SIZE 5

const int MAX_ITER = 10;


//MK: Averages each pixel's neighborhood on the CPU (baseline for comparison)
void cpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	for(int row=0; row<h; row++){
		for(int col=0; col<w; col++){
			float avgR = 0.0f;
			float avgG = 0.0f;
			float avgB = 0.0f;
			
			int pixels = 0;
			int index = 0;

			for(int rowOffset = -BLUR_SIZE; rowOffset < BLUR_SIZE+1; rowOffset++){
				for(int colOffset = -BLUR_SIZE; colOffset < BLUR_SIZE+1; colOffset++){
					int curRow = row + rowOffset;
					int curCol = col + colOffset;
	
					if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w){
						int curIndex = (curRow * w + curCol) * 3;
						avgR += inArray[curIndex];
						avgG += inArray[curIndex+1];
						avgB += inArray[curIndex+2];
						pixels++;
					}
				}
			}

			avgR = avgR / (float) pixels;
			avgG = avgG / (float) pixels;
			avgB = avgB / (float) pixels;

			index = (row * w + col) * 3;
			outArray[index] = (unsigned char) avgR;
			outArray[index+1] = (unsigned char) avgG;
			outArray[index+2] = (unsigned char) avgB;

		}
	}
}


//MK: TODO - average each pixel's neighborhood on the GPU, one thread per pixel (see cpuCode above)
__global__
void gpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
}

//MK: Main function
int main(){
	int w, h;
	unsigned char *h_imageArray;
	unsigned char *h_outImageArray;
	unsigned char *d_imageArray;
	unsigned char *d_outImageArray;
	unsigned char *h_outImageArray2;

	//MK: Load input image
	ppmLoad("../../DB/imageBlurDB/ewha_picture.ppm", &h_imageArray, &w, &h);

	size_t arraySize = sizeof(unsigned char) * h * w * 3;

	h_outImageArray = (unsigned char*)malloc(arraySize);
	h_outImageArray2 = (unsigned char*)malloc(arraySize);

	//MK: Allocate GPU memory
	cudaError_t err = cudaMalloc((void **) &d_imageArray, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_outImageArray, arraySize);
	checkCudaError(err);

	//MK: Copy input from host to GPU
	err = cudaMemcpy(d_imageArray, h_imageArray, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);

	const int tSize = 16;
	dim3 blockSize(tSize, tSize, 1);
	dim3 gridSize(ceil((float)w/tSize), ceil((float)h/tSize), 1);

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");
	clockMeasure *ckGpu = new clockMeasure("GPU CODE");

	ckCpu->clockReset();
	ckGpu->clockReset();

	for(int i = 0; i < MAX_ITER; i++){
		
		ckCpu->clockResume();
		cpuCode(h_outImageArray, h_imageArray, w, h);
		ckCpu->clockPause();

		ckGpu->clockResume();
		gpuCode<<<gridSize, blockSize>>>(d_outImageArray, d_imageArray, w, h);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);

	}

	ckCpu->clockPrint();
	ckGpu->clockPrint();

	//MK: Copy result from GPU back to host
	err = cudaMemcpy(h_outImageArray2, d_outImageArray, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	ppmSave("ewha_picture_cpu.ppm", h_outImageArray, w, h);
	ppmSave("ewha_picture_gpu.ppm", h_outImageArray2, w, h);

	cudaFree(d_imageArray);
	cudaFree(d_outImageArray);

	return 0;
}
