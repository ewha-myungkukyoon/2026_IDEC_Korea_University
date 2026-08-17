/*
 * =====================================================================================
 *
 *       Filename:  layer.h
 *
 *    Description:  Batched Conv/ReLU/Pool/FC layers for the simple
 *                  CIFAR-10 CNN: a CPU reference and a GPU kernel for
 *                  each, plus a runXxx() wrapper per USE_GPU_* toggle.
 *                  Every function takes a batchSize. All toggles are on
 *                  here, so the whole forward pass runs on the GPU.
 *
 *        Version:  1.0
 *        Created:  07/16/2026 05:00:00 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

#pragma once

//MK: All layers run on the GPU (see GPUEmpty/layer.h to bring them up one at a time instead)
#define USE_GPU_CONV1 1
#define USE_GPU_RELU  1
#define USE_GPU_POOL  1
#define USE_GPU_FC    1

const int THREADS = 256;

//MK: Ceiling-divide helper for 1D grid sizing
int numBlocks(int total){
	return (total + THREADS - 1) / THREADS;
}

//MK: ==================== CPU reference implementations (complete - read these while writing the GPU kernels below) ====================

//MK: Conv1 forward for a batch of images, no padding/stride 1 (weights are shared across the batch)
void conv1ForwardCpu(const float *input, const float *weight, const float *bias, float *output, int batchSize){
	for(int b = 0; b < batchSize; b++){
		const float *in = input + (size_t)b * CIFAR_IMAGE_SIZE;
		float *out = output + (size_t)b * CONV1_OUT_SIZE;

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
						float inVal = in[ic * IN_H * IN_W + ih * IN_W + iw];
						float wVal = weight[((oc * IN_C + ic) * CONV1_K + kh) * CONV1_K + kw];
						sum += inVal * wVal;
					}
				}
			}
			out[idx] = sum;
		}
	}
}

//MK: In-place ReLU over a whole batch at once (size = batchSize * CONV1_OUT_SIZE)
void reluForwardCpu(float *data, int size){
	for(int idx = 0; idx < size; idx++){
		data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
	}
}

//MK: 2x2 MaxPool forward for a batch of images
void poolForwardCpu(const float *input, float *output, int *argmax, int batchSize){
	for(int b = 0; b < batchSize; b++){
		const float *in = input + (size_t)b * CONV1_OUT_SIZE;
		float *out = output + (size_t)b * FLATTEN_SIZE;
		int *am = argmax + (size_t)b * FLATTEN_SIZE;

		for(int idx = 0; idx < FLATTEN_SIZE; idx++){
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
					float v = in[inIdx];
					if(v > maxVal){
						maxVal = v;
						maxIdx = inIdx;
					}
				}
			}
			out[idx] = maxVal;
			am[idx] = maxIdx;
		}
	}
}

//MK: FC1 forward for a batch of images (Flatten is implicit - pool output is already a contiguous 3600 vector per image)
void fcForwardCpu(const float *input, const float *weight, const float *bias, float *output, int batchSize){
	for(int b = 0; b < batchSize; b++){
		const float *in = input + (size_t)b * FLATTEN_SIZE;
		float *out = output + (size_t)b * NUM_CLASSES;

		for(int oc = 0; oc < NUM_CLASSES; oc++){
			float sum = bias[oc];
			for(int i = 0; i < FLATTEN_SIZE; i++){
				sum += weight[oc * FLATTEN_SIZE + i] * in[i];
			}
			out[oc] = sum;
		}
	}
}

//MK: ==================== GPU kernels, each matching its CPU counterpart above but for a whole batch ====================

//MK: Conv1 forward, one thread per (batch, output element) pair, no padding/stride 1
__global__
void conv1Forward(const float *input, const float *weight, const float *bias, float *output, int batchSize){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batchSize * CONV1_OUT_SIZE;
	if(idx >= total) return;

	int b = idx / CONV1_OUT_SIZE;
	int local = idx % CONV1_OUT_SIZE;
	const float *in = input + (size_t)b * CIFAR_IMAGE_SIZE;
	float *out = output + (size_t)b * CONV1_OUT_SIZE;

	int ow = local % CONV1_OUT_W;
	int oh = (local / CONV1_OUT_W) % CONV1_OUT_H;
	int oc = local / (CONV1_OUT_W * CONV1_OUT_H);

	float sum = bias[oc];
	for(int ic = 0; ic < IN_C; ic++){
		for(int kh = 0; kh < CONV1_K; kh++){
			for(int kw = 0; kw < CONV1_K; kw++){
				int ih = oh + kh;
				int iw = ow + kw;
				float inVal = in[ic * IN_H * IN_W + ih * IN_W + iw];
				float wVal = weight[((oc * IN_C + ic) * CONV1_K + kh) * CONV1_K + kw];
				sum += inVal * wVal;
			}
		}
	}
	out[local] = sum;
}

//MK: In-place ReLU, one thread per element (size already = batchSize * CONV1_OUT_SIZE, no batch decomposition needed)
__global__
void reluForward(float *data, int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

//MK: 2x2 MaxPool forward, one thread per (batch, pooled output element) pair
__global__
void poolForward(const float *input, float *output, int *argmax, int batchSize){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batchSize * FLATTEN_SIZE;
	if(idx >= total) return;

	int b = idx / FLATTEN_SIZE;
	int local = idx % FLATTEN_SIZE;
	const float *in = input + (size_t)b * CONV1_OUT_SIZE;
	float *out = output + (size_t)b * FLATTEN_SIZE;
	int *am = argmax + (size_t)b * FLATTEN_SIZE;

	int ow = local % POOL1_OUT_W;
	int oh = (local / POOL1_OUT_W) % POOL1_OUT_H;
	int oc = local / (POOL1_OUT_W * POOL1_OUT_H);

	int baseH = oh * POOL1_K;
	int baseW = ow * POOL1_K;

	float maxVal = -3.4e38f;
	int maxIdx = -1;
	for(int dh = 0; dh < POOL1_K; dh++){
		for(int dw = 0; dw < POOL1_K; dw++){
			int ih = baseH + dh;
			int iw = baseW + dw;
			int inIdx = oc * CONV1_OUT_H * CONV1_OUT_W + ih * CONV1_OUT_W + iw;
			float v = in[inIdx];
			if(v > maxVal){
				maxVal = v;
				maxIdx = inIdx;
			}
		}
	}
	out[local] = maxVal;
	am[local] = maxIdx;
}

//MK: FC1 forward, one thread per (batch, output class) pair (Flatten is implicit - pool output is already a contiguous 3600 vector per image)
__global__
void fcForward(const float *input, const float *weight, const float *bias, float *output, int batchSize){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batchSize * NUM_CLASSES;
	if(idx >= total) return;

	int b = idx / NUM_CLASSES;
	int oc = idx % NUM_CLASSES;
	const float *in = input + (size_t)b * FLATTEN_SIZE;
	float *out = output + (size_t)b * NUM_CLASSES;

	float sum = bias[oc];
	for(int i = 0; i < FLATTEN_SIZE; i++){
		sum += weight[oc * FLATTEN_SIZE + i] * in[i];
	}
	out[oc] = sum;
}

//MK: ==================== Device buffers, sized for up to BATCH_SIZE images at once ====================

float *d_bufImages, *d_bufConvOut, *d_bufPoolOut, *d_bufLogits;
int *d_bufPoolArgmax;
float *d_conv1W, *d_conv1Bias, *d_fc1W, *d_fc1Bias;

//MK: Allocates the device buffers; call once at startup, before the inference loop
void initGpuBuffers(){
	cudaError_t err;
	err = cudaMalloc((void**)&d_bufImages, (size_t)BATCH_SIZE * CIFAR_IMAGE_SIZE * sizeof(float));
	checkCudaError(err);
	err = cudaMalloc((void**)&d_bufConvOut, (size_t)BATCH_SIZE * CONV1_OUT_SIZE * sizeof(float));
	checkCudaError(err);
	err = cudaMalloc((void**)&d_bufPoolOut, (size_t)BATCH_SIZE * FLATTEN_SIZE * sizeof(float));
	checkCudaError(err);
	err = cudaMalloc((void**)&d_bufPoolArgmax, (size_t)BATCH_SIZE * FLATTEN_SIZE * sizeof(int));
	checkCudaError(err);
	err = cudaMalloc((void**)&d_bufLogits, (size_t)BATCH_SIZE * NUM_CLASSES * sizeof(float));
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
	cudaFree(d_bufImages);
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

//MK: ==================== Per-layer wrappers: GPU (whole batch, one launch) if enabled above, otherwise the CPU reference - main.cu only calls these ====================

//MK: h_conv1W/h_conv1Bias are only used on the CPU fallback path (the GPU path already has them uploaded via uploadWeights)
void runConv1(const float *h_input, const float *h_conv1W, const float *h_conv1Bias, float *h_output, int batchSize){
	if(USE_GPU_CONV1){
		cudaError_t err = cudaMemcpy(d_bufImages, h_input, (size_t)batchSize * CIFAR_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		conv1Forward<<<numBlocks(batchSize * CONV1_OUT_SIZE), THREADS>>>(d_bufImages, d_conv1W, d_conv1Bias, d_bufConvOut, batchSize);
		err = cudaDeviceSynchronize();
		checkCudaError(err);
		err = cudaMemcpy(h_output, d_bufConvOut, (size_t)batchSize * CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		checkCudaError(err);
	} else {
		conv1ForwardCpu(h_input, h_conv1W, h_conv1Bias, h_output, batchSize);
	}
}

void runRelu(float *h_data, int size){
	if(USE_GPU_RELU){
		cudaError_t err = cudaMemcpy(d_bufConvOut, h_data, (size_t)size * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		reluForward<<<numBlocks(size), THREADS>>>(d_bufConvOut, size);
		err = cudaDeviceSynchronize();
		checkCudaError(err);
		err = cudaMemcpy(h_data, d_bufConvOut, (size_t)size * sizeof(float), cudaMemcpyDeviceToHost);
		checkCudaError(err);
	} else {
		reluForwardCpu(h_data, size);
	}
}

void runPool(const float *h_input, float *h_output, int *h_argmax, int batchSize){
	if(USE_GPU_POOL){
		cudaError_t err = cudaMemcpy(d_bufConvOut, h_input, (size_t)batchSize * CONV1_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		poolForward<<<numBlocks(batchSize * FLATTEN_SIZE), THREADS>>>(d_bufConvOut, d_bufPoolOut, d_bufPoolArgmax, batchSize);
		err = cudaDeviceSynchronize();
		checkCudaError(err);
		err = cudaMemcpy(h_output, d_bufPoolOut, (size_t)batchSize * FLATTEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		checkCudaError(err);
		err = cudaMemcpy(h_argmax, d_bufPoolArgmax, (size_t)batchSize * FLATTEN_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
		checkCudaError(err);
	} else {
		poolForwardCpu(h_input, h_output, h_argmax, batchSize);
	}
}

//MK: h_fc1W/h_fc1Bias are only used on the CPU fallback path (the GPU path already has them uploaded via uploadWeights)
void runFc(const float *h_input, const float *h_fc1W, const float *h_fc1Bias, float *h_output, int batchSize){
	if(USE_GPU_FC){
		cudaError_t err = cudaMemcpy(d_bufPoolOut, h_input, (size_t)batchSize * FLATTEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		fcForward<<<numBlocks(batchSize * NUM_CLASSES), THREADS>>>(d_bufPoolOut, d_fc1W, d_fc1Bias, d_bufLogits, batchSize);
		err = cudaDeviceSynchronize();
		checkCudaError(err);
		err = cudaMemcpy(h_output, d_bufLogits, (size_t)batchSize * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
		checkCudaError(err);
	} else {
		fcForwardCpu(h_input, h_fc1W, h_fc1Bias, h_output, batchSize);
	}
}
