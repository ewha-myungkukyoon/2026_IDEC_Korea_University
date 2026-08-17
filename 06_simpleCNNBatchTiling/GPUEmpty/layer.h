/*
 * =====================================================================================
 *
 *       Filename:  layer.h
 *
 *    Description:  Exercise: reshape 05_simpleCNNBatchLowering's flat
 *                  gemm() into a classic square-tiled grid/block. No
 *                  __shared__ yet - that is 07. im2col() is unchanged;
 *                  only gemm()'s thread mapping changes. runGemm() already
 *                  launches a 2D block and a 2D grid for you.
 *
 *        Version:  1.0
 *        Created:  08/08/2026 10:00:00 AM
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

//MK: Lowering-specific shape constants, derived from the shared network-shape macros above
#define CONV1_K_DIM (IN_C * CONV1_K * CONV1_K)      //27 - length of one unfolded receptive-field column
#define CONV1_OUT_HW (CONV1_OUT_H * CONV1_OUT_W)     //900 - number of output spatial positions per image
#define GEMM_TILE CONV1_OUT_C                        //16 - output positions handled per GEMM block (so gemm()'s 2D blockDim = (GEMM_TILE, CONV1_OUT_C) = (16,16) = 256 threads)

//MK: ==================== CPU reference implementations (complete - read these while writing the GPU kernels below) ====================

//MK: im2col - unfold each image's 3x3 receptive fields into a (CONV1_K_DIM x CONV1_OUT_HW) column matrix, no padding/stride 1.
//MK: im2colMat layout is [batch][k][pos] so that, for a fixed (batch,k), consecutive pos are contiguous.
void im2colCpu(const float *input, float *im2colMat, int batchSize){
	for(int b = 0; b < batchSize; b++){
		const float *in = input + (size_t)b * CIFAR_IMAGE_SIZE;
		float *col = im2colMat + (size_t)b * CONV1_K_DIM * CONV1_OUT_HW;

		for(int k = 0; k < CONV1_K_DIM; k++){
			int kw = k % CONV1_K;
			int kh = (k / CONV1_K) % CONV1_K;
			int ic = k / (CONV1_K * CONV1_K);

			for(int pos = 0; pos < CONV1_OUT_HW; pos++){
				int ow = pos % CONV1_OUT_W;
				int oh = pos / CONV1_OUT_W;
				int ih = oh + kh;
				int iw = ow + kw;
				col[(size_t)k * CONV1_OUT_HW + pos] = in[ic * IN_H * IN_W + ih * IN_W + iw];
			}
		}
	}
}

//MK: GEMM - (CONV1_OUT_C x CONV1_K_DIM) weight times the (CONV1_K_DIM x CONV1_OUT_HW) im2col matrix, per image, plus bias.
//MK: weight is used exactly as trained (weight[oc*CONV1_K_DIM+k] == weight[((oc*IN_C+ic)*CONV1_K+kh)*CONV1_K+kw] from the direct-conv layout) - no reshaping needed.
//MK: output layout matches direct conv1Forward's [batch][channel][height][width], so reluForwardCpu/poolForwardCpu/fcForwardCpu below need no changes.
void matmulCpu(const float *im2colMat, const float *weight, const float *bias, float *output, int batchSize){
	for(int b = 0; b < batchSize; b++){
		const float *col = im2colMat + (size_t)b * CONV1_K_DIM * CONV1_OUT_HW;
		float *out = output + (size_t)b * CONV1_OUT_SIZE;

		for(int oc = 0; oc < CONV1_OUT_C; oc++){
			for(int pos = 0; pos < CONV1_OUT_HW; pos++){
				float sum = bias[oc];
				for(int k = 0; k < CONV1_K_DIM; k++){
					sum += weight[oc * CONV1_K_DIM + k] * col[(size_t)k * CONV1_OUT_HW + pos];
				}
				out[oc * CONV1_OUT_HW + pos] = sum;
			}
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

//MK: ==================== GPU kernels - TODO: implement each one to match its CPU counterpart above, but for a whole batch ====================

//MK: TODO - one thread per (batch, k, pos) triple (batchSize*CONV1_K_DIM*CONV1_OUT_HW total); see im2colCpu above.
//MK: Aim for consecutive threads to share (batch,k) and increment pos, so the input read and im2colMat write both coalesce.
__global__
void im2col(const float *input, float *im2colMat, int batchSize){
}

//MK: TODO - tiled GEMM, no shared memory yet; see matmulCpu above for the reference math.
//MK: runGemm() below launches with dim3 block(GEMM_TILE, CONV1_OUT_C) and dim3 grid(numTilesPerImage, batchSize):
//MK: blockIdx.x -> tile index, blockIdx.y -> batch index; threadIdx.x -> position-within-tile, threadIdx.y -> channel.
//MK: Each thread just reads its own weight row and im2col column straight from global memory (same access
//MK: pattern as 05_simpleCNNBatchLowering's gemm, just addressed via the tile/block indices) and writes its
//MK: one output element - no __shared__, no __syncthreads().
__global__
void gemm(const float *im2colMat, const float *weight, const float *bias, float *output, int batchSize){
}

//MK: TODO - one thread per element (size total, already batchSize*CONV1_OUT_SIZE - no batch decomposition needed); see reluForwardCpu above
__global__
void reluForward(float *data, int size){
}

//MK: TODO - one thread per (batch, pooled output element) pair (batchSize*FLATTEN_SIZE total); see poolForwardCpu above
__global__
void poolForward(const float *input, float *output, int *argmax, int batchSize){
}

//MK: TODO - one thread per (batch, output class) pair (batchSize*NUM_CLASSES total); see fcForwardCpu above
__global__
void fcForward(const float *input, const float *weight, const float *bias, float *output, int batchSize){
}

//MK: ==================== Device buffers, sized for up to BATCH_SIZE images at once ====================

float *d_bufImages, *d_bufIm2col, *d_bufConvOut, *d_bufPoolOut, *d_bufLogits;
int *d_bufPoolArgmax;
float *d_conv1W, *d_conv1Bias, *d_fc1W, *d_fc1Bias;

//MK: Host-side scratch for the CPU im2col+matmul fallback path (used whenever USE_GPU_CONV1 is 0)
float *h_bufIm2col;

//MK: Allocates the device buffers (and the CPU-path host scratch buffer); call once at startup, before the inference loop
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

	h_bufIm2col = (float*)malloc((size_t)BATCH_SIZE * CONV1_K_DIM * CONV1_OUT_HW * sizeof(float));

	if(USE_GPU_CONV1){
		err = cudaMalloc((void**)&d_bufIm2col, (size_t)BATCH_SIZE * CONV1_K_DIM * CONV1_OUT_HW * sizeof(float));
		checkCudaError(err);
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

//MK: Frees the device buffers (and the CPU-path host scratch buffer); call once after the inference loop
void freeGpuBuffers(){
	cudaFree(d_bufImages);
	cudaFree(d_bufConvOut);
	cudaFree(d_bufPoolOut);
	cudaFree(d_bufPoolArgmax);
	cudaFree(d_bufLogits);
	free(h_bufIm2col);
	if(USE_GPU_CONV1){
		cudaFree(d_bufIm2col);
		cudaFree(d_conv1W);
		cudaFree(d_conv1Bias);
	}
	if(USE_GPU_FC){
		cudaFree(d_fc1W);
		cudaFree(d_fc1Bias);
	}
}

//MK: ==================== Per-layer wrappers: GPU (whole batch, one launch) if enabled above, otherwise the CPU reference - main.cu only calls these ====================

//MK: im2col half of Conv1 - split out from gemm below (instead of one runConv1) so main.cu can time each half separately
void runIm2col(const float *h_input, int batchSize){
	if(USE_GPU_CONV1){
		cudaError_t err = cudaMemcpy(d_bufImages, h_input, (size_t)batchSize * CIFAR_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaError(err);
		im2col<<<numBlocks(batchSize * CONV1_K_DIM * CONV1_OUT_HW), THREADS>>>(d_bufImages, d_bufIm2col, batchSize);
		err = cudaDeviceSynchronize();
		checkCudaError(err);
	} else {
		im2colCpu(h_input, h_bufIm2col, batchSize);
	}
}

//MK: gemm half of Conv1 - reads the im2col matrix runIm2col above just produced (still on device for the GPU path, no extra transfer)
//MK: h_conv1W/h_conv1Bias are only used on the CPU fallback path (the GPU path already has them uploaded via uploadWeights)
void runGemm(const float *h_conv1W, const float *h_conv1Bias, float *h_output, int batchSize){
	if(USE_GPU_CONV1){
		int numTilesPerImage = (CONV1_OUT_HW + GEMM_TILE - 1) / GEMM_TILE;
		dim3 block(GEMM_TILE, CONV1_OUT_C);       //MK: (x,y) = (position within tile, output channel)
		dim3 grid(numTilesPerImage, batchSize);   //MK: (x,y) = (tile index, image index in the batch)
		gemm<<<grid, block>>>(d_bufIm2col, d_conv1W, d_conv1Bias, d_bufConvOut, batchSize);
		cudaError_t err = cudaDeviceSynchronize();
		checkCudaError(err);
		err = cudaMemcpy(h_output, d_bufConvOut, (size_t)batchSize * CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		checkCudaError(err);
	} else {
		matmulCpu(h_bufIm2col, h_conv1W, h_conv1Bias, h_output, batchSize);
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
