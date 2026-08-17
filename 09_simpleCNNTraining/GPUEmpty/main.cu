/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Exercise driver for training the CIFAR-10 CNN on the
 *                  GPU with mini-batch SGD. THIS FILE IS COMPLETE - the
 *                  exercise is layer.h's eight TODO kernels (backward +
 *                  optimizer). Verification runs before training and names
 *                  any wrong kernel. DB/ is never written to.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../common/clockMeasure.h"
#include "../../common/cudaCommon.h"
#include "../../common/cifar10.h"

//MK: All #define constants (network shapes + training config) live in common/simpleCnnConfig.h, shared with the CPU variant
#include "../../common/simpleCnnConfig.h"

//MK: Conv/ReLU/Pool/FC layers (GPU kernel + CPU counterpart pairs) - needs the shape macros from simpleCnnConfig.h
#include "layer.h"

//MK: This program trains its own weights and never reads or writes the pre-trained ones, but
//MK: simpleCnnUtil.h's verifyAccuracy needs loadExpectedResult declared, so the header still comes first
#include "../../common/simpleCnnWeight.h"

//MK: Uniform random init in [-scale, scale]
void initWeights(float *w, int size, float scale){
	for(int i = 0; i < size; i++){
		w[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
	}
}

//MK: Softmax + cross-entropy on the 10 logits (small enough to just do on the host, for both CPU and GPU paths)
float softmaxCrossEntropy(const float *logits, int label, float *dLogits){
	float maxLogit = logits[0];
	for(int i = 1; i < NUM_CLASSES; i++){
		if(logits[i] > maxLogit) maxLogit = logits[i];
	}

	float probs[NUM_CLASSES];
	float sumExp = 0.0f;
	for(int i = 0; i < NUM_CLASSES; i++){
		probs[i] = expf(logits[i] - maxLogit);
		sumExp += probs[i];
	}
	for(int i = 0; i < NUM_CLASSES; i++) probs[i] /= sumExp;

	for(int i = 0; i < NUM_CLASSES; i++){
		dLogits[i] = probs[i] - (i == label ? 1.0f : 0.0f);
	}
	return -logf(probs[label] + 1e-8f);
}

//MK: argmaxLogits/imageToFloat - needs NUM_CLASSES (simpleCnnConfig.h) and CIFAR_IMAGE_SIZE (cifar10.h)
#include "../../common/simpleCnnUtil.h"

//MK: verifyGpuKernels/compareTrainedModels - needs everything above (the layer kernel/CPU pairs, imageToFloat,
//MK: and softmaxCrossEntropy), so it has to come last
#include "../../common/simpleCnnVerify.h"

//MK: Fisher-Yates shuffle of an index array, so each epoch sees the training images (and mini-batch groupings) in a different order
void shuffleIndices(int *indices, int n){
	for(int i = n - 1; i > 0; i--){
		int j = rand() % (i + 1);
		int tmp = indices[i];
		indices[i] = indices[j];
		indices[j] = tmp;
	}
}

//MK: Main function
int main(){
	srand((unsigned int)time(NULL));

	//MK: Load CIFAR-10 train/test batches - all 5 training batches concatenated into one set
	unsigned char *trainLabels, *trainImages;
	unsigned char *testLabels, *testImages;
	int numTrain, numTest;
	const char *trainBatchFiles[] = {
		TRAIN_BATCH_FILE_1, TRAIN_BATCH_FILE_2, TRAIN_BATCH_FILE_3, TRAIN_BATCH_FILE_4, TRAIN_BATCH_FILE_5
	};
	cifar10LoadMultiple(trainBatchFiles, 5, &trainLabels, &trainImages, &numTrain);
	cifar10Load(TEST_BATCH_FILE, &testLabels, &testImages, &numTest);

	int trainCount = (TRAIN_IMAGES < numTrain) ? TRAIN_IMAGES : numTrain;
	int testCount = (TEST_IMAGES < numTest) ? TEST_IMAGES : numTest;

	//MK: Training image order - reshuffled every epoch so mini-batch groupings change too
	int *trainIndices = (int*)malloc(trainCount * sizeof(int));
	for(int i = 0; i < trainCount; i++) trainIndices[i] = i;

	//MK: Which implementation(s) to run this time (see RUN_MODE in simpleCnnConfig.h)
	bool runCpu = (RUN_MODE == RUN_CPU_ONLY || RUN_MODE == RUN_BOTH);
	bool runGpu = (RUN_MODE == RUN_GPU_ONLY || RUN_MODE == RUN_BOTH);
	printf("Run mode: %s\n", RUN_MODE == RUN_CPU_ONLY ? "CPU only" : RUN_MODE == RUN_GPU_ONLY ? "GPU only" : "CPU + GPU");

	//MK: Master initial weights - copied into both the CPU and GPU models so they start identically
	float *h_conv1W = (float*)malloc(CONV1_W_SIZE * sizeof(float));
	float *h_fc1W = (float*)malloc(FC1_W_SIZE * sizeof(float));
	initWeights(h_conv1W, CONV1_W_SIZE, WEIGHT_INIT_SCALE);
	initWeights(h_fc1W, FC1_W_SIZE, WEIGHT_INIT_SCALE);

	//MK: Biases start at zero (standard practice), not random
	float *h_conv1Bias = (float*)calloc(CONV1_OUT_C, sizeof(float));
	float *h_fc1Bias = (float*)calloc(NUM_CLASSES, sizeof(float));

	float *h_inputImage = (float*)malloc(CIFAR_IMAGE_SIZE * sizeof(float));
	float h_logits[NUM_CLASSES], h_dLogits[NUM_CLASSES];

	//MK: CPU model - its own weights, gradients, and activation buffers
	float *h_conv1W_cpu = (float*)malloc(CONV1_W_SIZE * sizeof(float));
	float *h_fc1W_cpu = (float*)malloc(FC1_W_SIZE * sizeof(float));
	float *h_conv1Bias_cpu = (float*)malloc(CONV1_OUT_C * sizeof(float));
	float *h_fc1Bias_cpu = (float*)malloc(NUM_CLASSES * sizeof(float));
	memcpy(h_conv1W_cpu, h_conv1W, CONV1_W_SIZE * sizeof(float));
	memcpy(h_fc1W_cpu, h_fc1W, FC1_W_SIZE * sizeof(float));
	memcpy(h_conv1Bias_cpu, h_conv1Bias, CONV1_OUT_C * sizeof(float));
	memcpy(h_fc1Bias_cpu, h_fc1Bias, NUM_CLASSES * sizeof(float));

	float *h_conv1dW_cpu = (float*)malloc(CONV1_W_SIZE * sizeof(float));
	float *h_fc1dW_cpu = (float*)malloc(FC1_W_SIZE * sizeof(float));
	float *h_conv1dBias_cpu = (float*)malloc(CONV1_OUT_C * sizeof(float));
	float *h_conv1Out_cpu = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));
	float *h_poolOut_cpu = (float*)malloc(FLATTEN_SIZE * sizeof(float));
	int *h_poolArgmax_cpu = (int*)malloc(FLATTEN_SIZE * sizeof(int));
	float *h_dPoolOut_cpu = (float*)malloc(FLATTEN_SIZE * sizeof(float));
	float *h_dConvOut_cpu = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));
	float h_logits_cpu[NUM_CLASSES], h_dLogits_cpu[NUM_CLASSES];

	//MK: CPU model - mini-batch gradient accumulators (summed over BATCH_SIZE images, then averaged into one SGD step)
	float *h_conv1dWBatch_cpu = (float*)malloc(CONV1_W_SIZE * sizeof(float));
	float *h_fc1dWBatch_cpu = (float*)malloc(FC1_W_SIZE * sizeof(float));
	float *h_conv1dBiasBatch_cpu = (float*)malloc(CONV1_OUT_C * sizeof(float));
	float *h_fc1dBiasBatch_cpu = (float*)malloc(NUM_CLASSES * sizeof(float));

	//MK: GPU model - weights/gradients plus one set of activation buffers reused every image (no batching of the forward/backward math itself)
	float *d_conv1W, *d_fc1W, *d_conv1dW, *d_fc1dW;
	float *d_conv1Bias, *d_fc1Bias, *d_conv1dBias;
	float *d_input, *d_conv1Out, *d_poolOut, *d_logits;
	float *d_dLogits, *d_dPoolOut, *d_dConvOut;
	int *d_poolArgmax;

	//MK: GPU model - mini-batch gradient accumulators
	float *d_conv1dWBatch, *d_fc1dWBatch, *d_conv1dBiasBatch, *d_fc1dBiasBatch;

	cudaError_t err;
	if(runGpu){
		err = cudaMalloc((void**)&d_conv1W, CONV1_W_SIZE * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_fc1W, FC1_W_SIZE * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_conv1dW, CONV1_W_SIZE * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_fc1dW, FC1_W_SIZE * sizeof(float)); checkCudaError(err);

		err = cudaMalloc((void**)&d_conv1Bias, CONV1_OUT_C * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_fc1Bias, NUM_CLASSES * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_conv1dBias, CONV1_OUT_C * sizeof(float)); checkCudaError(err);

		err = cudaMalloc((void**)&d_conv1dWBatch, CONV1_W_SIZE * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_fc1dWBatch, FC1_W_SIZE * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_conv1dBiasBatch, CONV1_OUT_C * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_fc1dBiasBatch, NUM_CLASSES * sizeof(float)); checkCudaError(err);

		err = cudaMalloc((void**)&d_input, CIFAR_IMAGE_SIZE * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_conv1Out, CONV1_OUT_SIZE * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_poolOut, FLATTEN_SIZE * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_logits, NUM_CLASSES * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_poolArgmax, FLATTEN_SIZE * sizeof(int)); checkCudaError(err);

		err = cudaMalloc((void**)&d_dLogits, NUM_CLASSES * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_dPoolOut, FLATTEN_SIZE * sizeof(float)); checkCudaError(err);
		err = cudaMalloc((void**)&d_dConvOut, CONV1_OUT_SIZE * sizeof(float)); checkCudaError(err);

		err = cudaMemcpy(d_conv1W, h_conv1W, CONV1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		err = cudaMemcpy(d_fc1W, h_fc1W, FC1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		err = cudaMemcpy(d_conv1Bias, h_conv1Bias, CONV1_OUT_C * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		err = cudaMemcpy(d_fc1Bias, h_fc1Bias, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
	}

#if VERIFY_KERNELS
	//MK: Before spending minutes on a training run, check every GPU kernel against its CPU counterpart on a few
	//MK: images. Each kernel is fed the CPU reference's inputs, so a failure names the one kernel that is wrong
	//MK: rather than every kernel downstream of it.
	if(runGpu){
		bool kernelsOk = verifyGpuKernels(h_conv1W, h_conv1Bias, h_fc1W, h_fc1Bias, trainImages, trainLabels, trainCount);
		if(!kernelsOk && VERIFY_STOP_ON_FAIL){
			printf("Stopping before training (set VERIFY_STOP_ON_FAIL to 0 in simpleCnnConfig.h to train anyway).\n");
			return 1;
		}
	}
#endif

	clockMeasure *ckCpuTrain = new clockMeasure("CPU TRAIN");
	clockMeasure *ckGpuTrain = new clockMeasure("GPU TRAIN");
	clockMeasure *ckCpuInfer = new clockMeasure("CPU INFER");
	clockMeasure *ckGpuInfer = new clockMeasure("GPU INFER");
	ckCpuTrain->clockReset();
	ckGpuTrain->clockReset();
	ckCpuInfer->clockReset();
	ckGpuInfer->clockReset();

	//MK: Training - mini-batch SGD (BATCH_SIZE images per step), still one image at a time through the forward/backward math, no optimizations; CPU and GPU run on the same (shuffled) data each iteration for a fair comparison
	for(int epoch = 0; epoch < EPOCHS; epoch++){
		float epochLossCpu = 0.0f;
		float epochLossGpu = 0.0f;
		int epochCorrectCpu = 0;
		int epochCorrectGpu = 0;

		shuffleIndices(trainIndices, trainCount);

		for(int batchStart = 0; batchStart < trainCount; batchStart += BATCH_SIZE){
			int batchEnd = (batchStart + BATCH_SIZE < trainCount) ? batchStart + BATCH_SIZE : trainCount;
			int batchSize = batchEnd - batchStart;

			if(runCpu){
				ckCpuTrain->clockResume();
				memset(h_conv1dWBatch_cpu, 0, CONV1_W_SIZE * sizeof(float));
				memset(h_fc1dWBatch_cpu, 0, FC1_W_SIZE * sizeof(float));
				memset(h_conv1dBiasBatch_cpu, 0, CONV1_OUT_C * sizeof(float));
				memset(h_fc1dBiasBatch_cpu, 0, NUM_CLASSES * sizeof(float));
				ckCpuTrain->clockPause();
			}
			if(runGpu){
				ckGpuTrain->clockResume();
				err = cudaMemset(d_conv1dWBatch, 0, CONV1_W_SIZE * sizeof(float)); checkCudaError(err);
				err = cudaMemset(d_fc1dWBatch, 0, FC1_W_SIZE * sizeof(float)); checkCudaError(err);
				err = cudaMemset(d_conv1dBiasBatch, 0, CONV1_OUT_C * sizeof(float)); checkCudaError(err);
				err = cudaMemset(d_fc1dBiasBatch, 0, NUM_CLASSES * sizeof(float)); checkCudaError(err);
				ckGpuTrain->clockPause();
			}

			for(int i = batchStart; i < batchEnd; i++){
				int idx = trainIndices[i];
				imageToFloat(trainImages + (size_t)idx * CIFAR_IMAGE_SIZE, h_inputImage);
				int label = trainLabels[idx];

				if(runCpu){
					//MK: CPU forward + backward, then accumulate into the batch gradient
					ckCpuTrain->clockResume();
					conv1ForwardCpu(h_inputImage, h_conv1W_cpu, h_conv1Bias_cpu, h_conv1Out_cpu);
					reluForwardCpu(h_conv1Out_cpu, CONV1_OUT_SIZE);
					poolForwardCpu(h_conv1Out_cpu, h_poolOut_cpu, h_poolArgmax_cpu);
					fcForwardCpu(h_poolOut_cpu, h_fc1W_cpu, h_fc1Bias_cpu, h_logits_cpu);

					float lossCpu = softmaxCrossEntropy(h_logits_cpu, label, h_dLogits_cpu);
					epochLossCpu += lossCpu;
					if(argmaxLogits(h_logits_cpu) == label) epochCorrectCpu++;

					fcBackwardWeightCpu(h_dLogits_cpu, h_poolOut_cpu, h_fc1dW_cpu);
					fcBackwardInputCpu(h_dLogits_cpu, h_fc1W_cpu, h_dPoolOut_cpu);
					memset(h_dConvOut_cpu, 0, CONV1_OUT_SIZE * sizeof(float));
					poolBackwardCpu(h_dPoolOut_cpu, h_poolArgmax_cpu, h_dConvOut_cpu);
					reluBackwardCpu(h_conv1Out_cpu, h_dConvOut_cpu, CONV1_OUT_SIZE);
					conv1BackwardWeightCpu(h_inputImage, h_dConvOut_cpu, h_conv1dW_cpu);
					conv1BackwardBiasCpu(h_dConvOut_cpu, h_conv1dBias_cpu);

					accumulateGradCpu(h_conv1dWBatch_cpu, h_conv1dW_cpu, CONV1_W_SIZE);
					accumulateGradCpu(h_fc1dWBatch_cpu, h_fc1dW_cpu, FC1_W_SIZE);
					accumulateGradCpu(h_conv1dBiasBatch_cpu, h_conv1dBias_cpu, CONV1_OUT_C);
					accumulateGradCpu(h_fc1dBiasBatch_cpu, h_dLogits_cpu, NUM_CLASSES);
					ckCpuTrain->clockPause();
				}

				if(runGpu){
					//MK: GPU forward + backward, then accumulate into the batch gradient
					ckGpuTrain->clockResume();
					err = cudaMemcpy(d_input, h_inputImage, CIFAR_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
					checkCudaError(err);

					conv1Forward<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(d_input, d_conv1W, d_conv1Bias, d_conv1Out);
					reluForward<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(d_conv1Out, CONV1_OUT_SIZE);
					poolForward<<<numBlocks(FLATTEN_SIZE), THREADS>>>(d_conv1Out, d_poolOut, d_poolArgmax);
					fcForward<<<numBlocks(NUM_CLASSES), THREADS>>>(d_poolOut, d_fc1W, d_fc1Bias, d_logits);

					err = cudaMemcpy(h_logits, d_logits, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
					checkCudaError(err);

					float lossGpu = softmaxCrossEntropy(h_logits, label, h_dLogits);
					epochLossGpu += lossGpu;
					if(argmaxLogits(h_logits) == label) epochCorrectGpu++;

					err = cudaMemcpy(d_dLogits, h_dLogits, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);
					checkCudaError(err);

					fcBackwardWeight<<<numBlocks(FC1_W_SIZE), THREADS>>>(d_dLogits, d_poolOut, d_fc1dW);
					fcBackwardInput<<<numBlocks(FLATTEN_SIZE), THREADS>>>(d_dLogits, d_fc1W, d_dPoolOut);

					err = cudaMemset(d_dConvOut, 0, CONV1_OUT_SIZE * sizeof(float));
					checkCudaError(err);
					poolBackward<<<numBlocks(FLATTEN_SIZE), THREADS>>>(d_dPoolOut, d_poolArgmax, d_dConvOut);
					reluBackward<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(d_conv1Out, d_dConvOut, CONV1_OUT_SIZE);
					conv1BackwardWeight<<<numBlocks(CONV1_W_SIZE), THREADS>>>(d_input, d_dConvOut, d_conv1dW);
					conv1BackwardBias<<<numBlocks(CONV1_OUT_C), THREADS>>>(d_dConvOut, d_conv1dBias);

					accumulateGrad<<<numBlocks(CONV1_W_SIZE), THREADS>>>(d_conv1dWBatch, d_conv1dW, CONV1_W_SIZE);
					accumulateGrad<<<numBlocks(FC1_W_SIZE), THREADS>>>(d_fc1dWBatch, d_fc1dW, FC1_W_SIZE);
					accumulateGrad<<<numBlocks(CONV1_OUT_C), THREADS>>>(d_conv1dBiasBatch, d_conv1dBias, CONV1_OUT_C);
					accumulateGrad<<<numBlocks(NUM_CLASSES), THREADS>>>(d_fc1dBiasBatch, d_dLogits, NUM_CLASSES);

					err = cudaDeviceSynchronize();
					checkCudaError(err);
					ckGpuTrain->clockPause();
				}

				if((i + 1) % PRINT_EVERY == 0){
					if(runCpu && runGpu){
						printf("Epoch %d, Image %d/%d, Avg Loss So Far (CPU %f / GPU %f), Train Acc So Far (CPU %.2f%% / GPU %.2f%%)\n",
								epoch, i + 1, trainCount, epochLossCpu / (i + 1), epochLossGpu / (i + 1),
								100.0f * epochCorrectCpu / (i + 1), 100.0f * epochCorrectGpu / (i + 1));
					} else if(runCpu){
						printf("Epoch %d, Image %d/%d, Avg Loss So Far (CPU %f), Train Acc So Far (CPU %.2f%%)\n",
								epoch, i + 1, trainCount, epochLossCpu / (i + 1), 100.0f * epochCorrectCpu / (i + 1));
					} else {
						printf("Epoch %d, Image %d/%d, Avg Loss So Far (GPU %f), Train Acc So Far (GPU %.2f%%)\n",
								epoch, i + 1, trainCount, epochLossGpu / (i + 1), 100.0f * epochCorrectGpu / (i + 1));
					}
				}
			}

			//MK: One averaged SGD step per batch (dividing by batchSize via the effective learning rate)
			float batchLR = LEARNING_RATE / batchSize;
			if(runCpu){
				ckCpuTrain->clockResume();
				sgdUpdateCpu(h_conv1W_cpu, h_conv1dWBatch_cpu, batchLR, CONV1_W_SIZE);
				sgdUpdateCpu(h_fc1W_cpu, h_fc1dWBatch_cpu, batchLR, FC1_W_SIZE);
				sgdUpdateCpu(h_conv1Bias_cpu, h_conv1dBiasBatch_cpu, batchLR, CONV1_OUT_C);
				sgdUpdateCpu(h_fc1Bias_cpu, h_fc1dBiasBatch_cpu, batchLR, NUM_CLASSES);
				ckCpuTrain->clockPause();
			}
			if(runGpu){
				ckGpuTrain->clockResume();
				sgdUpdate<<<numBlocks(CONV1_W_SIZE), THREADS>>>(d_conv1W, d_conv1dWBatch, batchLR, CONV1_W_SIZE);
				sgdUpdate<<<numBlocks(FC1_W_SIZE), THREADS>>>(d_fc1W, d_fc1dWBatch, batchLR, FC1_W_SIZE);
				sgdUpdate<<<numBlocks(CONV1_OUT_C), THREADS>>>(d_conv1Bias, d_conv1dBiasBatch, batchLR, CONV1_OUT_C);
				sgdUpdate<<<numBlocks(NUM_CLASSES), THREADS>>>(d_fc1Bias, d_fc1dBiasBatch, batchLR, NUM_CLASSES);
				err = cudaDeviceSynchronize();
				checkCudaError(err);
				ckGpuTrain->clockPause();
			}
		}

		if(runCpu && runGpu){
			printf("== Epoch %d done, Avg Loss (CPU %f / GPU %f), Train Acc (CPU %.2f%% / GPU %.2f%%) ==\n", epoch,
					epochLossCpu / trainCount, epochLossGpu / trainCount,
					100.0f * epochCorrectCpu / trainCount, 100.0f * epochCorrectGpu / trainCount);
		} else if(runCpu){
			printf("== Epoch %d done, Avg Loss (CPU %f), Train Acc (CPU %.2f%%) ==\n", epoch,
					epochLossCpu / trainCount, 100.0f * epochCorrectCpu / trainCount);
		} else {
			printf("== Epoch %d done, Avg Loss (GPU %f), Train Acc (GPU %.2f%%) ==\n", epoch,
					epochLossGpu / trainCount, 100.0f * epochCorrectGpu / trainCount);
		}
	}

	printf("\n=== Training ===\n");
	if(runCpu) ckCpuTrain->clockPrint();
	if(runGpu) ckGpuTrain->clockPrint();

	//MK: The trained weights stop here, in memory. There is no writer anywhere in these examples (see
	//MK: simpleCnnWeight.h): the pre-trained DB/03_simpleCNN-weight/weights.bin that 03 through 08 load,
	//MK: and the expected_result.txt next to it, must stay exactly as shipped.

	//MK: Inference - forward pass only, each model evaluated with its own trained weights
	int correctCpu = 0, correctGpu = 0;
	for(int i = 0; i < testCount; i++){
		imageToFloat(testImages + (size_t)i * CIFAR_IMAGE_SIZE, h_inputImage);
		int label = testLabels[i];

		if(runCpu){
			ckCpuInfer->clockResume();
			conv1ForwardCpu(h_inputImage, h_conv1W_cpu, h_conv1Bias_cpu, h_conv1Out_cpu);
			reluForwardCpu(h_conv1Out_cpu, CONV1_OUT_SIZE);
			poolForwardCpu(h_conv1Out_cpu, h_poolOut_cpu, h_poolArgmax_cpu);
			fcForwardCpu(h_poolOut_cpu, h_fc1W_cpu, h_fc1Bias_cpu, h_logits_cpu);
			ckCpuInfer->clockPause();
			if(argmaxLogits(h_logits_cpu) == label) correctCpu++;
		}

		if(runGpu){
			ckGpuInfer->clockResume();
			err = cudaMemcpy(d_input, h_inputImage, CIFAR_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
			checkCudaError(err);

			conv1Forward<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(d_input, d_conv1W, d_conv1Bias, d_conv1Out);
			reluForward<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(d_conv1Out, CONV1_OUT_SIZE);
			poolForward<<<numBlocks(FLATTEN_SIZE), THREADS>>>(d_conv1Out, d_poolOut, d_poolArgmax);
			fcForward<<<numBlocks(NUM_CLASSES), THREADS>>>(d_poolOut, d_fc1W, d_fc1Bias, d_logits);

			err = cudaMemcpy(h_logits, d_logits, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
			checkCudaError(err);
			ckGpuInfer->clockPause();
			if(argmaxLogits(h_logits) == label) correctGpu++;
		}
	}

	printf("\n=== Inference ===\n");
	if(runCpu){
		printf("CPU Top-1 Test Accuracy: %d/%d (%.2f%%)\n", correctCpu, testCount, 100.0f * correctCpu / testCount);
		ckCpuInfer->clockPrint();
	}
	if(runGpu){
		printf("GPU Top-1 Test Accuracy: %d/%d (%.2f%%)\n", correctGpu, testCount, 100.0f * correctGpu / testCount);
		ckGpuInfer->clockPrint();
		//MK: This accuracy is only printed, never recorded. DB/03_simpleCNN-weight/expected_result.txt belongs
		//MK: to the pre-trained weights.bin next to it, and the two must keep describing the same model
	}

#if VERIFY_TRAINED_MODEL
	//MK: Did the two independently trained models actually learn the same thing? Needs both to have trained
	if(runCpu && runGpu){
		//MK: Pull the GPU's final weights down - nothing else copies them back, so this is the only way the
		//MK: host-side comparison below can see what the GPU model actually learned
		err = cudaMemcpy(h_conv1W, d_conv1W, CONV1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		err = cudaMemcpy(h_conv1Bias, d_conv1Bias, CONV1_OUT_C * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		err = cudaMemcpy(h_fc1W, d_fc1W, FC1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		err = cudaMemcpy(h_fc1Bias, d_fc1Bias, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);

		compareTrainedModels(h_conv1W_cpu, h_conv1Bias_cpu, h_fc1W_cpu, h_fc1Bias_cpu,
				h_conv1W, h_conv1Bias, h_fc1W, h_fc1Bias,
				correctCpu, correctGpu, testCount);
	} else {
		printf("\n(Trained model comparison skipped - needs RUN_MODE = RUN_BOTH in simpleCnnConfig.h)\n");
	}
#endif

	//MK: Cleanup
	free(trainLabels); free(trainImages); free(trainIndices);
	free(testLabels); free(testImages);
	free(h_conv1W); free(h_fc1W); free(h_conv1Bias); free(h_fc1Bias); free(h_inputImage);
	free(h_conv1W_cpu); free(h_fc1W_cpu); free(h_conv1Bias_cpu); free(h_fc1Bias_cpu);
	free(h_conv1dW_cpu); free(h_fc1dW_cpu); free(h_conv1dBias_cpu);
	free(h_conv1dWBatch_cpu); free(h_fc1dWBatch_cpu); free(h_conv1dBiasBatch_cpu); free(h_fc1dBiasBatch_cpu);
	free(h_conv1Out_cpu); free(h_poolOut_cpu); free(h_poolArgmax_cpu);
	free(h_dPoolOut_cpu); free(h_dConvOut_cpu);

	if(runGpu){
		cudaFree(d_conv1W); cudaFree(d_fc1W); cudaFree(d_conv1dW); cudaFree(d_fc1dW);
		cudaFree(d_conv1Bias); cudaFree(d_fc1Bias); cudaFree(d_conv1dBias);
		cudaFree(d_conv1dWBatch); cudaFree(d_fc1dWBatch); cudaFree(d_conv1dBiasBatch); cudaFree(d_fc1dBiasBatch);
		cudaFree(d_input); cudaFree(d_conv1Out); cudaFree(d_poolOut); cudaFree(d_logits);
		cudaFree(d_dLogits); cudaFree(d_dPoolOut); cudaFree(d_dConvOut); cudaFree(d_poolArgmax);
	}

	return 0;
}
