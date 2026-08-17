/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Exercise: add two float vectors on the GPU and compare
 *                  the result against the CPU implementation, measuring
 *                  the execution time of each. The CPU side is already
 *                  written - fill in each //MK: TODO below, in order, to
 *                  build up the GPU side.
 *
 *        Version:  1.0
 *        Created:  08/16/2026 01:45:00 PM
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

//MK: TODO - add two vectors on the GPU, one thread per element (see cpuVectorAddition above)
__global__
void gpuVectorAddition(float *d_a, float *d_b, float *d_c, const int size){
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
	//MK: TODO - cudaMalloc d_a, d_b, d_c (arraySize bytes each) and check each with checkCudaError()

	//MK: Copy input from host to GPU
	//MK: TODO - cudaMemcpy inputA into d_a and inputB into d_b (cudaMemcpyHostToDevice)

	//MK: TODO - set gridSize/blockSize: tSize threads per block, enough blocks to cover MAX_SIZE
	const int tSize = 256;

	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuVectorAddition(inputA, inputB, cpuAns, MAX_SIZE);
		ckCpu->clockPause();

		ckGpu->clockResume();
		//MK: TODO - launch gpuVectorAddition, then cudaDeviceSynchronize() before the timer stops
		ckGpu->clockPause();
	}

	//MK: Copy result from GPU back to host
	//MK: TODO - cudaMemcpy d_c into gpuAns (cudaMemcpyDeviceToHost)

	//MK: TODO - cudaFree d_a, d_b, d_c

	checkAnswer(cpuAns, gpuAns, MAX_SIZE);

	ckCpu->clockPrint();
	ckGpu->clockPrint();

	return 0;
}
