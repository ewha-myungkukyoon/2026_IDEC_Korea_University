/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Exercise: implement the empty kernels below for one
 *                  large convolution layer - conv2dDirect, im2col,
 *                  castToHalf, biasAdd, and four GEMMs of the same product
 *                  with progressively more structure (flat -> 2D tiled ->
 *                  + shared -> Tensor Core). Rerun to turn [FAIL] green.
 *
 *        Version:  1.0
 *        Created:  08/11/2026 12:00:00 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../../common/clockMeasure.h"
#include "../../common/cudaCommon.h"

//MK: for wmma:: (Tensor Core WMMA API) below
using namespace nvcuda;

//MK: ==================== Conv layer shape ====================
#define CONV1_IN_C   256
#define CONV1_OUT_C  256
#define CONV1_K      3      //MK: 3x3 kernel
#define CONV1_STRIDE 1
#define CONV1_PAD    1
#define INPUT_H      56
#define INPUT_W      56
#define OUTPUT_H     56     //MK: (INPUT_H + 2*PAD - K)/STRIDE + 1 = 56, i.e. "same" padding
#define OUTPUT_W     56

#define INPUT_SIZE     (CONV1_IN_C * INPUT_H * INPUT_W)                  //MK: 802,816 elements  (~3.2 MB)
#define CONV1_W_SIZE   (CONV1_OUT_C * CONV1_IN_C * CONV1_K * CONV1_K)    //MK: 589,824 elements  (~2.4 MB)
#define CONV1_OUT_HW   (OUTPUT_H * OUTPUT_W)                             //MK: 3,136  - im2col N (output positions)
#define CONV1_K_DIM    (CONV1_IN_C * CONV1_K * CONV1_K)                  //MK: 2,304  - im2col K (reduction dim)
#define CONV1_OUT_SIZE (CONV1_OUT_C * CONV1_OUT_HW)                      //MK: 802,816 elements  (~3.2 MB) - im2col M*N
#define IM2COL_SIZE    ((size_t)CONV1_K_DIM * CONV1_OUT_HW)              //MK: 7,225,344 elements (~29 MB)

#define GEMM_TILE 16   //MK: 256, 2304, 3136 are all multiples of 16 -> no boundary padding needed anywhere below

#define NUM_TRIALS 10       //MK: each implementation below is timed over this many runs
#define VERIFY_EPSILON 1e-3 //MK: max allowed abs diff against the CPU reference for a GPU result to "pass"

//MK: Tensor Core input operands are cast to fp16 (~3 decimal digits of precision) before the K=2304-term
//MK: accumulation below, so its result needs a looser tolerance than the pure-fp32 paths above even though
//MK: the accumulator itself stays fp32.
#define VERIFY_EPSILON_HALF 5e-3

const int THREADS = 256;

//MK: Ceiling-divide helper for 1D grid sizing
int numBlocks(int total){
	return (total + THREADS - 1) / THREADS;
}

//MK: ==================== Random init ====================

//MK: Fixed seed so every run sees the exact same input/weight/bias values
void randomInit(float *data, size_t n, float scale){
	for(size_t i = 0; i < n; i++){
		data[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
	}
}

//MK: ==================== CPU reference: direct convolution with same padding ====================

void conv2dDirectCpu(const float *input, const float *weight, const float *bias, float *output){
	for(int idx = 0; idx < CONV1_OUT_SIZE; idx++){
		int ow = idx % OUTPUT_W;
		int oh = (idx / OUTPUT_W) % OUTPUT_H;
		int oc = idx / (OUTPUT_W * OUTPUT_H);

		float sum = bias[oc];
		for(int ic = 0; ic < CONV1_IN_C; ic++){
			for(int kh = 0; kh < CONV1_K; kh++){
				int ih = oh * CONV1_STRIDE - CONV1_PAD + kh;
				if(ih < 0 || ih >= INPUT_H) continue;
				for(int kw = 0; kw < CONV1_K; kw++){
					int iw = ow * CONV1_STRIDE - CONV1_PAD + kw;
					if(iw < 0 || iw >= INPUT_W) continue;
					float inVal = input[ic * INPUT_H * INPUT_W + ih * INPUT_W + iw];
					float wVal = weight[((oc * CONV1_IN_C + ic) * CONV1_K + kh) * CONV1_K + kw];
					sum += inVal * wVal;
				}
			}
		}
		output[idx] = sum;
	}
}

//MK: ==================== GPU kernels ====================

//MK: TODO - one thread per output pixel (CONV1_OUT_SIZE total); port conv2dDirectCpu above straight across,
//MK: replacing its outer `for(idx...)` loop with `idx = blockIdx.x * blockDim.x + threadIdx.x` and a bounds check.
__global__
void conv2dDirect(const float *input, const float *weight, const float *bias, float *output){
	//MK: TODO - students implement this kernel
}

//MK: TODO - one thread per (k,pos) pair (CONV1_K_DIM*CONV1_OUT_HW total), zero-padded at the input border.
//MK: im2colMat layout is [k][pos] so that, for a fixed k, consecutive pos are contiguous: im2colMat[k*CONV1_OUT_HW+pos].
//MK: Decode idx -> pos = idx % CONV1_OUT_HW, k = idx / CONV1_OUT_HW, then k -> (ic,kh,kw) and pos -> (oh,ow)
//MK: the same way conv2dDirectCpu above decodes its (oc,oh,ow)/(ic,kh,kw) indices, and write 0 for any (ih,iw)
//MK: that falls outside the padded input.
__global__
void im2col(const float *input, float *im2colMat){
	//MK: TODO - students implement this kernel
}

//MK: TODO - GEMM #1, flat/naive ("lowering"): one thread per (oc,pos) output element (CONV1_OUT_SIZE total,
//MK: launched with numBlocks(CONV1_OUT_SIZE)/THREADS, same 1D scheme as conv2dDirect above). For your (oc,pos)
//MK: pair, sum bias[oc] plus a straight dot product over the full K=CONV1_K_DIM reduction dimension:
//MK: sum += weight[oc*CONV1_K_DIM+k] * im2colMat[k*CONV1_OUT_HW+pos] for k in [0,CONV1_K_DIM).
__global__
void gemmLowering(const float *im2colMat, const float *weight, const float *bias, float *output){
	//MK: TODO - students implement this kernel
}

//MK: TODO - GEMM #2, tiled grid/block, still no shared memory: computes the exact same sum as gemmLowering
//MK: above, but launched with a 2D grid/block (gemmGrid/gemmBlock, both GEMM_TILE x GEMM_TILE = 16x16) instead
//MK: of a flat 1D one, so each block owns a 16-positions x 16-channels tile of the output. Derive your output
//MK: channel/position from blockIdx/threadIdx instead of a flat idx: oc = blockIdx.y*GEMM_TILE + threadIdx.y,
//MK: pos = blockIdx.x*GEMM_TILE + threadIdx.x. Each thread still reads its own weight/im2col elements straight
//MK: from global memory every iteration - no __shared__ yet, that's the next kernel below.
__global__
void gemmTiled(const float *im2colMat, const float *weight, const float *bias, float *output){
	//MK: TODO - students implement this kernel
}

//MK: TODO - GEMM #3, tiled + shared memory: same 16x16 block/tile layout as gemmTiled above (same oc/pos
//MK: derivation from blockIdx/threadIdx), but for each K_DIM chunk of GEMM_TILE, first cooperatively stage
//MK: that chunk's weight row and im2col column into two __shared__ float[GEMM_TILE][GEMM_TILE] tiles (one
//MK: element per thread each - every one of the 256 threads in the block loads exactly one weight value and
//MK: one im2col value), __syncthreads(), then do the GEMM_TILE multiply-adds reading from the shared tiles
//MK: instead of global memory, __syncthreads() again before moving to the next chunk.
__global__
void gemmTiledShared(const float *im2colMat, const float *weight, const float *bias, float *output){
	//MK: TODO - students implement this kernel
}

//MK: ==================== Tensor Core GEMM (WMMA, fp16 input / fp32 accumulate) ====================

//MK: TODO - one-off fp32->fp16 cast, reused for both weight and im2colMat: one thread per element (n total,
//MK: same flat 1D scheme as conv2dDirect above), dst[idx] = __float2half(src[idx]). Tensor Cores need
//MK: half-precision operands; the accumulator stays fp32 (see gemmTensorCore below), so this truncation is
//MK: the only source of extra numerical error in that path - which is why it verifies against the looser
//MK: VERIFY_EPSILON_HALF instead of VERIFY_EPSILON.
__global__
void castToHalf(const float *src, half *dst, int n){
	//MK: TODO - students implement this kernel
}

//MK: TODO - GEMM #4, Tensor Core via WMMA: same 16x16 output tile and same gemmGrid as gemmTiled/gemmTiledShared
//MK: above, but launched with only 32 threads (blockDim=32, i.e. ONE warp) per block instead of 256 - the whole
//MK: warp cooperatively computes the entire 16x16 tile in hardware, so there is no per-thread output element here.
//MK: Declare three fragments with wmma::fragment<> (matrix_a / matrix_b / accumulator, all 16x16x16 = GEMM_TILE,
//MK: half operands, float accumulator), zero the accumulator with wmma::fill_fragment, then walk K in GEMM_TILE
//MK: steps: wmma::load_matrix_sync each operand tile (weight row-major with leading dimension CONV1_K_DIM,
//MK: im2colMat row-major with leading dimension CONV1_OUT_HW), and wmma::mma_sync to multiply-accumulate.
//MK: Finish with wmma::store_matrix_sync into output (leading dimension CONV1_OUT_HW, wmma::mem_row_major).
//MK: No bias here - the accumulator fragment has no documented per-element -> (row,col) mapping, so bias is
//MK: added by the separate biasAdd pass below.
__global__
void gemmTensorCore(const half *im2colMat, const half *weight, float *output){
	//MK: TODO - students implement this kernel
}

//MK: TODO - add bias in a separate pass: one thread per output element (CONV1_OUT_SIZE total, same flat 1D
//MK: scheme as conv2dDirect above), oc = idx / CONV1_OUT_HW, output[idx] += bias[oc].
__global__
void biasAdd(float *output, const float *bias){
	//MK: TODO - students implement this kernel
}

//MK: ==================== Host helper: verify a GPU result against the CPU reference ====================

//MK: Returns 1 (pass) if every element is within epsilon of the CPU reference, else 0 (fail) - epsilon
//MK: defaults to VERIFY_EPSILON, but the Tensor Core path below passes VERIFY_EPSILON_HALF explicitly
int verifyResult(const char *label, const float *ref, const float *test, int n, double epsilon = VERIFY_EPSILON){
	double maxDiff = 0.0;
	for(int i = 0; i < n; i++){
		double d = fabs((double)ref[i] - (double)test[i]);
		if(d > maxDiff) maxDiff = d;
	}
	int pass = (maxDiff <= epsilon);
	printf("\t[%s] %-28s maxAbsDiff=%.6f\n", pass ? "PASS" : "FAIL", label, maxDiff);
	return pass;
}

//MK: Main function
int main(){
	srand(42);

	//MK: ==================== Host memory allocation ====================
	float *h_input         = (float*)malloc(INPUT_SIZE * sizeof(float));
	float *h_weight        = (float*)malloc(CONV1_W_SIZE * sizeof(float));
	float *h_bias          = (float*)malloc(CONV1_OUT_C * sizeof(float));
	float *h_cpu           = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));  //MK: CPU reference output
	float *h_conv          = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));  //MK: GPU direct conv output
	float *h_lowering      = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));  //MK: GPU im2col+naive GEMM output
	float *h_tiling        = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));  //MK: GPU im2col+tiled GEMM output
	float *h_tiling_shared = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));  //MK: GPU im2col+tiled+shared GEMM output
	float *h_tensor_core   = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));  //MK: GPU im2col+Tensor Core GEMM output

	//MK: ==================== Random number generation ====================
	randomInit(h_input, INPUT_SIZE, 0.05f);
	randomInit(h_weight, CONV1_W_SIZE, 0.05f);
	randomInit(h_bias, CONV1_OUT_C, 0.05f);

	//MK: ==================== Device memory allocation ====================
	float *d_input, *d_weight, *d_bias;
	float *d_conv, *d_lowering, *d_tiling, *d_tiling_shared, *d_tensor_core, *d_im2col;
	half *d_weight_half, *d_im2col_half;  //MK: fp16 copies of d_weight/d_im2col, fed to gemmTensorCore below
	cudaError_t err;
	err = cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float));           checkCudaError(err);
	err = cudaMalloc((void**)&d_weight, CONV1_W_SIZE * sizeof(float));        checkCudaError(err);
	err = cudaMalloc((void**)&d_bias, CONV1_OUT_C * sizeof(float));           checkCudaError(err);
	err = cudaMalloc((void**)&d_conv, CONV1_OUT_SIZE * sizeof(float));        checkCudaError(err);
	err = cudaMalloc((void**)&d_lowering, CONV1_OUT_SIZE * sizeof(float));    checkCudaError(err);
	err = cudaMalloc((void**)&d_tiling, CONV1_OUT_SIZE * sizeof(float));      checkCudaError(err);
	err = cudaMalloc((void**)&d_tiling_shared, CONV1_OUT_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&d_tensor_core, CONV1_OUT_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&d_im2col, IM2COL_SIZE * sizeof(float));         checkCudaError(err);
	err = cudaMalloc((void**)&d_weight_half, CONV1_W_SIZE * sizeof(half));    checkCudaError(err);
	err = cudaMalloc((void**)&d_im2col_half, IM2COL_SIZE * sizeof(half));     checkCudaError(err);

	//MK: ==================== Copy random input/weight/bias from host to device ====================
	err = cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);     checkCudaError(err);
	err = cudaMemcpy(d_weight, h_weight, CONV1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice);  checkCudaError(err);
	err = cudaMemcpy(d_bias, h_bias, CONV1_OUT_C * sizeof(float), cudaMemcpyHostToDevice);       checkCudaError(err);

	printf("=== Large Conv Layer Benchmark ===\n");
	printf("Conv:   C_in=%d H=W=%d K=%dx%d stride=%d pad=%d C_out=%d -> Hout=Wout=%d\n",
	       CONV1_IN_C, INPUT_H, CONV1_K, CONV1_K, CONV1_STRIDE, CONV1_PAD, CONV1_OUT_C, OUTPUT_H);
	printf("GEMM:   M=%d (C_out)  K=%d (C_in*Kh*Kw)  N=%d (Hout*Wout)  GEMM_TILE=%d\n",
	       CONV1_OUT_C, CONV1_K_DIM, CONV1_OUT_HW, GEMM_TILE);
	printf("Trials: %d\n\n", NUM_TRIALS);

	clockMeasure *ckCpu          = new clockMeasure("CPU direct conv");
	clockMeasure *ckConv         = new clockMeasure("GPU direct conv");
	clockMeasure *ckIm2col       = new clockMeasure("im2col");
	clockMeasure *ckLowering     = new clockMeasure("GPU im2col+naive GEMM");
	clockMeasure *ckTiling       = new clockMeasure("GPU im2col+tiled GEMM");
	clockMeasure *ckTilingShared = new clockMeasure("GPU im2col+tiled+shared GEMM");
	clockMeasure *ckCastHalf     = new clockMeasure("fp16 cast (weight+im2col)");
	clockMeasure *ckTensorCore   = new clockMeasure("GPU im2col+Tensor Core GEMM");
	ckCpu->clockReset(); ckConv->clockReset(); ckIm2col->clockReset();
	ckLowering->clockReset(); ckTiling->clockReset(); ckTilingShared->clockReset();
	ckCastHalf->clockReset(); ckTensorCore->clockReset();

	//MK: Tiled/shared GEMM kernels (gemmTiled, gemmTiledShared) use a 2D thread block: threadIdx.y selects the
	//MK: output channel (oc) within the tile, threadIdx.x selects the output position (pos) within the tile -
	//MK: this maps 1:1 onto the GEMM_TILE x GEMM_TILE output tile each block computes. The direct-conv/naive-GEMM
	//MK: kernels above don't tile, so they keep a flat 1D thread index instead.
	dim3 gemmBlock(GEMM_TILE, GEMM_TILE);
	dim3 gemmGrid(CONV1_OUT_HW / GEMM_TILE, CONV1_OUT_C / GEMM_TILE);

	//MK: ==================== Benchmark loop ====================
	for(int t = 0; t < NUM_TRIALS; t++){
		//MK: CPU
		ckCpu->clockResume();
		conv2dDirectCpu(h_input, h_weight, h_bias, h_cpu);
		ckCpu->clockPause();

		//MK: GPU_conv
		ckConv->clockResume();
		conv2dDirect<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(d_input, d_weight, d_bias, d_conv);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(h_conv, d_conv, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		ckConv->clockPause();

		//MK: im2col - input is identical every trial, so this only needs to run once
		if(t == 0){
			ckIm2col->clockResume();
			im2col<<<numBlocks(CONV1_K_DIM * CONV1_OUT_HW), THREADS>>>(d_input, d_im2col);
			err = cudaDeviceSynchronize(); checkCudaError(err);
			ckIm2col->clockPause();
		}

		//MK: GPU_lowering
		ckLowering->clockResume();
		gemmLowering<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(d_im2col, d_weight, d_bias, d_lowering);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(h_lowering, d_lowering, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		ckLowering->clockPause();

		//MK: GPU_tiling
		ckTiling->clockResume();
		gemmTiled<<<gemmGrid, gemmBlock>>>(d_im2col, d_weight, d_bias, d_tiling);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(h_tiling, d_tiling, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		ckTiling->clockPause();

		//MK: GPU_tiling + shared memory
		ckTilingShared->clockResume();
		gemmTiledShared<<<gemmGrid, gemmBlock>>>(d_im2col, d_weight, d_bias, d_tiling_shared);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(h_tiling_shared, d_tiling_shared, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		ckTilingShared->clockPause();

		//MK: ==================== fp16 cast (weight+im2col) - input never changes, so this only needs to run once ====================
		if(t == 0){
			ckCastHalf->clockResume();
			castToHalf<<<numBlocks(CONV1_W_SIZE), THREADS>>>(d_weight, d_weight_half, CONV1_W_SIZE);
			castToHalf<<<numBlocks((int)IM2COL_SIZE), THREADS>>>(d_im2col, d_im2col_half, (int)IM2COL_SIZE);
			err = cudaDeviceSynchronize(); checkCudaError(err);
			ckCastHalf->clockPause();
		}

		//MK: GPU im2col + Tensor Core GEMM (WMMA, fp16 in / fp32 accumulate) + separate bias-add pass
		ckTensorCore->clockResume();
		gemmTensorCore<<<gemmGrid, 32>>>(d_im2col_half, d_weight_half, d_tensor_core);
		biasAdd<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(d_tensor_core, d_bias);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(h_tensor_core, d_tensor_core, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		ckTensorCore->clockPause();
	}

	ckCpu->clockPrint();
	ckConv->clockPrint();
	ckIm2col->clockPrint();
	ckLowering->clockPrint();
	ckTiling->clockPrint();
	ckTilingShared->clockPrint();
	ckCastHalf->clockPrint();
	ckTensorCore->clockPrint();
	printf("\n");

	//MK: ==================== Verification (compare every GPU result against the CPU reference) ====================
	printf("=== Verification (vs CPU reference, epsilon=%.g) ===\n", VERIFY_EPSILON);
	int allPass = 1;
	allPass &= verifyResult("GPU direct conv", h_cpu, h_conv, CONV1_OUT_SIZE);
	allPass &= verifyResult("GPU im2col+naive GEMM", h_cpu, h_lowering, CONV1_OUT_SIZE);
	allPass &= verifyResult("GPU im2col+tiled GEMM", h_cpu, h_tiling, CONV1_OUT_SIZE);
	allPass &= verifyResult("GPU im2col+tiled+shared GEMM", h_cpu, h_tiling_shared, CONV1_OUT_SIZE);
	allPass &= verifyResult("GPU im2col+Tensor Core GEMM", h_cpu, h_tensor_core, CONV1_OUT_SIZE, VERIFY_EPSILON_HALF);
	printf("\n");

	//MK: ==================== Summary ====================
	if(allPass){
		double cpuAvg      = ckCpu->getAvgTimeMs();
		double convAvg     = ckConv->getAvgTimeMs();
		double im2colMs    = ckIm2col->getAvgTimeMs();  //MK: single measurement (count==1)
		double loweringAvg = ckLowering->getAvgTimeMs() + im2colMs;
		double tilingAvg   = ckTiling->getAvgTimeMs() + im2colMs;
		double sharedAvg   = ckTilingShared->getAvgTimeMs() + im2colMs;
		double castHalfMs  = ckCastHalf->getAvgTimeMs();    //MK: single measurement (count==1)
		double tensorCoreAvg = ckTensorCore->getAvgTimeMs() + im2colMs + castHalfMs;

		printf("All results PASSED verification.\n\n");
		printf("=== Summary (avg per run, %d trials; im2col/fp16-cast time folded into the GEMM rows below) ===\n", NUM_TRIALS);
		printf("%-32s %12s %14s\n", "Implementation", "Avg (ms)", "Speedup/CPU");
		printf("%-32s %12.3f %14.2f\n", "CPU direct conv", cpuAvg, 1.0);
		printf("%-32s %12.3f %14.2f\n", "GPU direct conv", convAvg, cpuAvg / convAvg);
		printf("%-32s %12.3f %14.2f\n", "GPU im2col+naive GEMM", loweringAvg, cpuAvg / loweringAvg);
		printf("%-32s %12.3f %14.2f\n", "GPU im2col+tiled GEMM", tilingAvg, cpuAvg / tilingAvg);
		printf("%-32s %12.3f %14.2f\n", "GPU im2col+tiled+shared GEMM", sharedAvg, cpuAvg / sharedAvg);
		printf("%-32s %12.3f %14.2f\n", "GPU im2col+Tensor Core GEMM", tensorCoreAvg, cpuAvg / tensorCoreAvg);
	} else {
		printf("Verification FAILED - see [FAIL] entries above for the mismatching implementation(s).\n");
	}

	//MK: Cleanup
	free(h_input); free(h_weight); free(h_bias); free(h_cpu);
	free(h_conv); free(h_lowering); free(h_tiling); free(h_tiling_shared); free(h_tensor_core);
	cudaFree(d_input); cudaFree(d_weight); cudaFree(d_bias);
	cudaFree(d_conv); cudaFree(d_lowering); cudaFree(d_tiling); cudaFree(d_tiling_shared); cudaFree(d_im2col);
	cudaFree(d_tensor_core); cudaFree(d_weight_half); cudaFree(d_im2col_half);

	return 0;
}
