/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Adds two float vectors on the GPU and compares the
 *                  result against a CPU implementation, measuring the
 *                  execution time of each.
 *
 *        Version:  1.0
 *        Created:  07/14/2026 01:02:00 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon 
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

const int MAX_SIZE=100000;
const float MAX_NUM=100.0;
const int MAX_ITER= 1000;

#include <stdio.h>
#include "../../common/clockMeasure.h"
#include "../../common/cudaCommon.h"

float inputA[MAX_SIZE];
float inputB[MAX_SIZE];
float gpuAns[MAX_SIZE];
float cpuAns[MAX_SIZE];

//MK: Fills an array with random float values in [0, max)
void generateRandomValues(float *array, float max, const int size){
	for(int i = 0; i < size; i++){
		array[i] = float(rand())/float(RAND_MAX) * max;
	}
}

//MK: Adds two vectors on the CPU (baseline for comparison)
void cpuVectorAddition(float *h_a, float *h_b, float *h_c, const int size){
	for(int i = 0; i < size; i++){
		h_c[i] = h_a[i] + h_b[i];
	}
}

//MK: Adds two vectors on the GPU, one thread per element
__global__
void gpuVectorAddition(float *d_a, float *d_b, float *d_c, const int size){
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < size){
		d_c[tId] = d_a[tId] + d_b[tId];
	}
}

//MK: Compares CPU and GPU results element-by-element
void checkAnswer(float *h_a, float *d_a, const int size){
	bool isSame = true;
	for(int i = 0; i < size; i++){
		if(h_a[i] != d_a[i]){
			printf("-\tERROR: IDX - %d ( %f != %f )\n", i, h_a[i], d_a[i]);
			isSame = false;
		}
	}
	if(isSame)
		printf("All values are same\n");
	else
		printf("Some values are not same\n");
}

//MK: Main function
int main(){
	srand((unsigned int)time(NULL));

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");
	ckCpu->clockReset();
	clockMeasure *ckGpu = new clockMeasure("GPU CODE");
	ckGpu->clockReset();
	
	//MK: Random input generation
	generateRandomValues(inputA, MAX_NUM, MAX_SIZE);
	generateRandomValues(inputB, MAX_NUM, MAX_SIZE);

	//MK: Allocate GPU memory
	float *d_a, *d_b, *d_c;
	int arraySize = MAX_SIZE * sizeof(float);
	cudaError_t err = cudaMalloc((void **) &d_a, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_b, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_c, arraySize);
	checkCudaError(err);

	//MK: Copy input from host to GPU
	err = cudaMemcpy(d_a, inputA, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);
	err = cudaMemcpy(d_b, inputB, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);

	const int tSize = 256;
	dim3 gridSize(ceil((float)MAX_SIZE/(float)tSize), 1, 1);
	dim3 blockSize(tSize, 1, 1);
	
	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuVectorAddition(inputA, inputB, cpuAns, MAX_SIZE);
		ckCpu->clockPause();
		
		ckGpu->clockResume();
		gpuVectorAddition<<<gridSize, blockSize>>>(d_a, d_b, d_c, MAX_SIZE);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);
	}

	//MK: Copy result from GPU back to host
	err = cudaMemcpy(gpuAns, d_c, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	checkAnswer(cpuAns, gpuAns, MAX_SIZE);

	ckCpu->clockPrint();
	ckGpu->clockPrint();

	return 0;
}
