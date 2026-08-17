/*
 * =====================================================================================
 *
 *       Filename:  layer.h
 *
 *    Description:  Exercise: port the simple CIFAR-10 CNN's forward pass
 *                  to the GPU one layer at a time. Each layer's CPU
 *                  reference is written - implement the __global__ kernel
 *                  next to it and flip its USE_GPU_* toggle on, in order.
 *                  Accuracy must not change; the timing should.
 *
 *        Version:  1.0
 *        Created:  07/16/2026 03:00:00 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

#pragma once

//MK: Toggle which layers run on the GPU - flip these on one at a time, in order, as you implement each kernel below
#define USE_GPU_CONV1 0
#define USE_GPU_RELU  0
#define USE_GPU_POOL  0
#define USE_GPU_FC    0

const int THREADS = 256;

//MK: Ceiling-divide helper for 1D grid sizing
int numBlocks(int total){
	return (total + THREADS - 1) / THREADS;
}

//MK: ==================== CPU reference implementations (complete - read these while writing the GPU kernels below) ====================

//MK: Conv1 forward, one plain loop over every output element, no padding/stride 1
void conv1ForwardCpu(const float *input, const float *weight, const float *bias, float *output){
	for(int idx = 0; idx < CONV1_OUT_SIZE; idx++){
		int ow = idx % CONV1_OUT_W;
		int oh = (idx / CONV1_OUT_W) % CONV1_OUT_H;
		int oc = idx / (CONV1_OUT_W * CONV1_OUT_H);

		float sum = bias[oc];
		for(int ic = 0; ic < IN_C; ic++){
			for(int kh = 0; kh < CONV1_K; kh++){
				for(int kw = 0; kw < CONV1_K; kw++){
					int ih = oh + kh;
					int iw = ow + kw;
					float inVal = input[ic * IN_H * IN_W + ih * IN_W + iw];
					float wVal = weight[((oc * IN_C + ic) * CONV1_K + kh) * CONV1_K + kw];
					sum += inVal * wVal;
				}
			}
		}
		output[idx] = sum;
	}
}

//MK: In-place ReLU
void reluForwardCpu(float *data, int size){
	for(int idx = 0; idx < size; idx++){
		data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
	}
}

//MK: 2x2 MaxPool forward
void poolForwardCpu(const float *input, float *output, int *argmax){
	int total = CONV1_OUT_C * POOL1_OUT_H * POOL1_OUT_W;
	for(int idx = 0; idx < total; idx++){
		int ow = idx % POOL1_OUT_W;
		int oh = (idx / POOL1_OUT_W) % POOL1_OUT_H;
		int oc = idx / (POOL1_OUT_W * POOL1_OUT_H);

		int baseH = oh * POOL1_K;
		int baseW = ow * POOL1_K;

		float maxVal = -3.4e38f;
		int maxIdx = -1;
		for(int dh = 0; dh < POOL1_K; dh++){
			for(int dw = 0; dw < POOL1_K; dw++){
				int ih = baseH + dh;
				int iw = baseW + dw;
				int inIdx = oc * CONV1_OUT_H * CONV1_OUT_W + ih * CONV1_OUT_W + iw;
				float v = input[inIdx];
				if(v > maxVal){
					maxVal = v;
					maxIdx = inIdx;
				}
			}
		}
		output[idx] = maxVal;
		argmax[idx] = maxIdx;
	}
}

//MK: FC1 forward (Flatten is implicit - pool output is already a contiguous 3600 vector)
void fcForwardCpu(const float *input, const float *weight, const float *bias, float *output){
	for(int oc = 0; oc < NUM_CLASSES; oc++){
		float sum = bias[oc];
		for(int i = 0; i < FLATTEN_SIZE; i++){
			sum += weight[oc * FLATTEN_SIZE + i] * input[i];
		}
		output[oc] = sum;
	}
}

//MK: ==================== GPU kernels - TODO: implement each one to match its CPU counterpart above ====================

//MK: TODO - one thread per output element (CONV1_OUT_SIZE total); see conv1ForwardCpu above
__global__
void conv1Forward(const float *input, const float *weight, const float *bias, float *output){
}

//MK: TODO - one thread per element (size total); see reluForwardCpu above
__global__
void reluForward(float *data, int size){
}

//MK: TODO - one thread per pooled output element (FLATTEN_SIZE total); see poolForwardCpu above
__global__
void poolForward(const float *input, float *output, int *argmax){
}

//MK: TODO - one thread per output class (NUM_CLASSES total); see fcForwardCpu above
__global__
void fcForward(const float *input, const float *weight, const float *bias, float *output){
}

//MK: ==================== Device buffers shared by whichever layers are toggled on above ====================

float *d_bufImage, *d_bufConvOut, *d_bufPoolOut, *d_bufLogits;
int *d_bufPoolArgmax;
float *d_conv1W, *d_conv1Bias, *d_fc1W, *d_fc1Bias;

//MK: Allocates the device buffers; call once at startup, before the inference loop
void initGpuBuffers(){
	cudaError_t err;
	err = cudaMalloc((void**)&d_bufImage, CIFAR_IMAGE_SIZE * sizeof(float));
	checkCudaError(err);
	err = cudaMalloc((void**)&d_bufConvOut, CONV1_OUT_SIZE * sizeof(float));
	checkCudaError(err);
	err = cudaMalloc((void**)&d_bufPoolOut, FLATTEN_SIZE * sizeof(float));
	checkCudaError(err);
	err = cudaMalloc((void**)&d_bufPoolArgmax, FLATTEN_SIZE * sizeof(int));
	checkCudaError(err);
	err = cudaMalloc((void**)&d_bufLogits, NUM_CLASSES * sizeof(float));
	checkCudaError(err);

	if(USE_GPU_CONV1){
		err = cudaMalloc((void**)&d_conv1W, CONV1_W_SIZE * sizeof(float));
		checkCudaError(err);
		err = cudaMalloc((void**)&d_conv1Bias, CONV1_OUT_C * sizeof(float));
		checkCudaError(err);
	}
	if(USE_GPU_FC){
		err = cudaMalloc((void**)&d_fc1W, FC1_W_SIZE * sizeof(float));
		checkCudaError(err);
		err = cudaMalloc((void**)&d_fc1Bias, NUM_CLASSES * sizeof(float));
		checkCudaError(err);
	}
}

//MK: Uploads the already-loaded host weights to the device, for whichever layers are GPU-enabled; call once after loadWeights
void uploadWeights(const float *h_conv1W, const float *h_conv1Bias, const float *h_fc1W, const float *h_fc1Bias){
	cudaError_t err;
	if(USE_GPU_CONV1){
		err = cudaMemcpy(d_conv1W, h_conv1W, CONV1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		err = cudaMemcpy(d_conv1Bias, h_conv1Bias, CONV1_OUT_C * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
	}
	if(USE_GPU_FC){
		err = cudaMemcpy(d_fc1W, h_fc1W, FC1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		err = cudaMemcpy(d_fc1Bias, h_fc1Bias, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
	}
}

//MK: Frees the device buffers; call once after the inference loop
void freeGpuBuffers(){
	cudaFree(d_bufImage);
	cudaFree(d_bufConvOut);
	cudaFree(d_bufPoolOut);
	cudaFree(d_bufPoolArgmax);
	cudaFree(d_bufLogits);
	if(USE_GPU_CONV1){
		cudaFree(d_conv1W);
		cudaFree(d_conv1Bias);
	}
	if(USE_GPU_FC){
		cudaFree(d_fc1W);
		cudaFree(d_fc1Bias);
	}
}

//MK: ==================== Per-layer wrappers: GPU if enabled above, otherwise the CPU reference - main.cu only calls these ====================

//MK: h_conv1W/h_conv1Bias are only used on the CPU fallback path (the GPU path already has them uploaded via uploadWeights)
void runConv1(const float *h_input, const float *h_conv1W, const float *h_conv1Bias, float *h_output){
	if(USE_GPU_CONV1){
		cudaError_t err = cudaMemcpy(d_bufImage, h_input, CIFAR_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		conv1Forward<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(d_bufImage, d_conv1W, d_conv1Bias, d_bufConvOut);
		err = cudaDeviceSynchronize();
		checkCudaError(err);
		err = cudaMemcpy(h_output, d_bufConvOut, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		checkCudaError(err);
	} else {
		conv1ForwardCpu(h_input, h_conv1W, h_conv1Bias, h_output);
	}
}

void runRelu(float *h_data, int size){
	if(USE_GPU_RELU){
		cudaError_t err = cudaMemcpy(d_bufConvOut, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		reluForward<<<numBlocks(size), THREADS>>>(d_bufConvOut, size);
		err = cudaDeviceSynchronize();
		checkCudaError(err);
		err = cudaMemcpy(h_data, d_bufConvOut, size * sizeof(float), cudaMemcpyDeviceToHost);
		checkCudaError(err);
	} else {
		reluForwardCpu(h_data, size);
	}
}

void runPool(const float *h_input, float *h_output, int *h_argmax){
	if(USE_GPU_POOL){
		cudaError_t err = cudaMemcpy(d_bufConvOut, h_input, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		poolForward<<<numBlocks(FLATTEN_SIZE), THREADS>>>(d_bufConvOut, d_bufPoolOut, d_bufPoolArgmax);
		err = cudaDeviceSynchronize();
		checkCudaError(err);
		err = cudaMemcpy(h_output, d_bufPoolOut, FLATTEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		checkCudaError(err);
		err = cudaMemcpy(h_argmax, d_bufPoolArgmax, FLATTEN_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
		checkCudaError(err);
	} else {
		poolForwardCpu(h_input, h_output, h_argmax);
	}
}

//MK: h_fc1W/h_fc1Bias are only used on the CPU fallback path (the GPU path already has them uploaded via uploadWeights)
void runFc(const float *h_input, const float *h_fc1W, const float *h_fc1Bias, float *h_output){
	if(USE_GPU_FC){
		cudaError_t err = cudaMemcpy(d_bufPoolOut, h_input, FLATTEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		fcForward<<<numBlocks(NUM_CLASSES), THREADS>>>(d_bufPoolOut, d_fc1W, d_fc1Bias, d_bufLogits);
		err = cudaDeviceSynchronize();
		checkCudaError(err);
		err = cudaMemcpy(h_output, d_bufLogits, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
		checkCudaError(err);
	} else {
		fcForwardCpu(h_input, h_fc1W, h_fc1Bias, h_output);
	}
}
