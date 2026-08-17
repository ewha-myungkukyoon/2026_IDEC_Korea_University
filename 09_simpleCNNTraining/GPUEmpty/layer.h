/*
 * =====================================================================================
 *
 *       Filename:  layer.h
 *
 *    Description:  Exercise: port the CIFAR-10 CNN's BACKWARD pass and
 *                  optimizer to the GPU. The four forward kernels are
 *                  already filled in (identical to 03) - write the eight
 *                  kernels marked TODO, each mirroring the CPU reference
 *                  directly above it. Run after each one to see PASS/FAIL.
 *
 *        Version:  1.0
 *        Created:  08/15/2026 11:30:00 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

#pragma once

const int THREADS = 256;

//MK: Ceiling-divide helper for 1D grid sizing - launch with <<<numBlocks(total), THREADS>>>
int numBlocks(int total){
	return (total + THREADS - 1) / THREADS;
}

//MK: ==================== Forward pass (given - identical to 03_simpleCNNInference/GPU/layer.h) ====================

//MK: CPU reference: one plain loop over every output element, no padding/stride 1
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

//MK: Conv1 forward, one thread per output element, no padding/stride 1
__global__
void conv1Forward(const float *input, const float *weight, const float *bias, float *output){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= CONV1_OUT_SIZE) return;

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

//MK: CPU reference: in-place ReLU
void reluForwardCpu(float *data, int size){
	for(int idx = 0; idx < size; idx++){
		data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
	}
}

//MK: In-place ReLU, one thread per element
__global__
void reluForward(float *data, int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

//MK: CPU reference: 2x2 MaxPool, recording the winning input index for backward
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

//MK: 2x2 MaxPool forward, one thread per pooled output; records the winning index for backward
__global__
void poolForward(const float *input, float *output, int *argmax){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = CONV1_OUT_C * POOL1_OUT_H * POOL1_OUT_W;
	if(idx >= total) return;

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

//MK: CPU reference (Flatten is implicit - pool output is already a contiguous 3600 vector)
void fcForwardCpu(const float *input, const float *weight, const float *bias, float *output){
	for(int oc = 0; oc < NUM_CLASSES; oc++){
		float sum = bias[oc];
		for(int i = 0; i < FLATTEN_SIZE; i++){
			sum += weight[oc * FLATTEN_SIZE + i] * input[i];
		}
		output[oc] = sum;
	}
}

//MK: FC1 forward (Flatten is implicit - pool output is already a contiguous 3600 vector)
__global__
void fcForward(const float *input, const float *weight, const float *bias, float *output){
	int oc = blockIdx.x * blockDim.x + threadIdx.x;
	if(oc >= NUM_CLASSES) return;

	float sum = bias[oc];
	for(int i = 0; i < FLATTEN_SIZE; i++){
		sum += weight[oc * FLATTEN_SIZE + i] * input[i];
	}
	output[oc] = sum;
}

//MK: ==================== 1. FC1 backward: weight gradient ====================

//MK: CPU reference: dW_fc1[oc][i] = dLogits[oc] * flattenInput[i]
void fcBackwardWeightCpu(const float *dLogits, const float *input, float *dWeight){
	for(int idx = 0; idx < FC1_W_SIZE; idx++){
		int i = idx % FLATTEN_SIZE;
		int oc = idx / FLATTEN_SIZE;
		dWeight[idx] = dLogits[oc] * input[i];
	}
}

//MK: TODO - one thread per weight (FC1_W_SIZE total); one multiply each, no loop at all
__global__
void fcBackwardWeight(const float *dLogits, const float *input, float *dWeight){
}

//MK: ==================== 2. FC1 backward: input gradient ====================

//MK: CPU reference: dFlatten[i] = sum_oc dLogits[oc] * W_fc1[oc][i]
void fcBackwardInputCpu(const float *dLogits, const float *weight, float *dInput){
	for(int i = 0; i < FLATTEN_SIZE; i++){
		float sum = 0.0f;
		for(int oc = 0; oc < NUM_CLASSES; oc++){
			sum += dLogits[oc] * weight[oc * FLATTEN_SIZE + i];
		}
		dInput[i] = sum;
	}
}

//MK: TODO - one thread per flatten element (FLATTEN_SIZE total); the oc loop stays as a loop
__global__
void fcBackwardInput(const float *dLogits, const float *weight, float *dInput){
}

//MK: ==================== 3. MaxPool backward ====================

//MK: CPU reference: scatters dFlatten back to the position each pooled max came from (dReluOut must be zeroed first)
void poolBackwardCpu(const float *dPoolOut, const int *argmax, float *dReluOut){
	int total = CONV1_OUT_C * POOL1_OUT_H * POOL1_OUT_W;
	for(int idx = 0; idx < total; idx++){
		dReluOut[argmax[idx]] = dPoolOut[idx];
	}
}

//MK: TODO - one thread per pooled element (FLATTEN_SIZE total). Note the write is indirect - the thread writes
//MK: to argmax[idx], not to idx. No atomicAdd is needed here: 2x2 windows never overlap, so no two threads
//MK: target the same address. The caller zeroes dReluOut first, since the losers never get written at all
__global__
void poolBackward(const float *dPoolOut, const int *argmax, float *dReluOut){
}

//MK: ==================== 4. ReLU backward ====================

//MK: CPU reference: zeroes the gradient wherever the forward activation was clamped to 0
void reluBackwardCpu(const float *reluOut, float *dConvOut, int size){
	for(int idx = 0; idx < size; idx++){
		if(reluOut[idx] <= 0.0f) dConvOut[idx] = 0.0f;
	}
}

//MK: TODO - one thread per element (size total); reluOut is the ReLU *output*, which is exactly 0 where it clamped
__global__
void reluBackward(const float *reluOut, float *dConvOut, int size){
}

//MK: ==================== 5. Conv1 backward: weight gradient ====================

//MK: CPU reference: one weight at a time, summed over every output position that used it
void conv1BackwardWeightCpu(const float *input, const float *dConvOut, float *dWeight){
	for(int idx = 0; idx < CONV1_W_SIZE; idx++){
		int kw = idx % CONV1_K;
		int kh = (idx / CONV1_K) % CONV1_K;
		int ic = (idx / (CONV1_K * CONV1_K)) % IN_C;
		int oc = idx / (CONV1_K * CONV1_K * IN_C);

		float sum = 0.0f;
		for(int oh = 0; oh < CONV1_OUT_H; oh++){
			for(int ow = 0; ow < CONV1_OUT_W; ow++){
				int ih = oh + kh;
				int iw = ow + kw;
				float inVal = input[ic * IN_H * IN_W + ih * IN_W + iw];
				float dOutVal = dConvOut[oc * CONV1_OUT_H * CONV1_OUT_W + oh * CONV1_OUT_W + ow];
				sum += inVal * dOutVal;
			}
		}
		dWeight[idx] = sum;
	}
}

//MK: TODO - one thread per weight (CONV1_W_SIZE total); the oh/ow loops stay as loops. Each thread owns one
//MK: weight and accumulates into a local sum, so no atomics are needed - that ownership is the whole point
__global__
void conv1BackwardWeight(const float *input, const float *dConvOut, float *dWeight){
}

//MK: ==================== 6. Conv1 backward: bias gradient ====================

//MK: CPU reference: dBias_conv1[oc] = sum of dConvOut over all output positions of that channel
void conv1BackwardBiasCpu(const float *dConvOut, float *dBias){
	for(int oc = 0; oc < CONV1_OUT_C; oc++){
		float sum = 0.0f;
		for(int oh = 0; oh < CONV1_OUT_H; oh++){
			for(int ow = 0; ow < CONV1_OUT_W; ow++){
				sum += dConvOut[oc * CONV1_OUT_H * CONV1_OUT_W + oh * CONV1_OUT_W + ow];
			}
		}
		dBias[oc] = sum;
	}
}

//MK: TODO - one thread per output channel (CONV1_OUT_C total); the oh/ow loops stay as loops
__global__
void conv1BackwardBias(const float *dConvOut, float *dBias){
}

//MK: ==================== 7. Mini-batch gradient accumulation ====================

//MK: CPU reference: accum[i] += grad[i] - sums per-image gradients into a mini-batch total
void accumulateGradCpu(float *accum, const float *grad, int size){
	for(int idx = 0; idx < size; idx++){
		accum[idx] += grad[idx];
	}
}

//MK: TODO - one thread per element (size total)
__global__
void accumulateGrad(float *accum, const float *grad, int size){
}

//MK: ==================== 8. SGD update ====================

//MK: CPU reference: plain SGD step, reused for both the conv1 and fc1 weight/bias buffers
void sgdUpdateCpu(float *weight, const float *grad, float lr, int size){
	for(int idx = 0; idx < size; idx++){
		weight[idx] -= lr * grad[idx];
	}
}

//MK: TODO - one thread per element (size total); lr already carries the 1/batchSize averaging, see main.cu
__global__
void sgdUpdate(float *weight, const float *grad, float lr, int size){
}
