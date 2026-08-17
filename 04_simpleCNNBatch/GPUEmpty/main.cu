/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Exercise: port the BATCHED CIFAR-10 CNN inference to
 *                  the GPU one layer at a time, BATCH_SIZE images per
 *                  kernel launch. Compare the timing against
 *                  03_simpleCNNInference's per-image numbers. Work in
 *                  layer.h - this file needs no changes.
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

#include <stdio.h>
#include <stdlib.h>
#include "../../common/clockMeasure.h"
#include "../../common/cifar10.h"
#include "../../common/cudaCommon.h"

//MK: All #define constants (network shapes + training config, including BATCH_SIZE) live in common/simpleCnnConfig.h
#include "../../common/simpleCnnConfig.h"

//MK: loadWeights - needs the shape macros from simpleCnnConfig.h
#include "../../common/simpleCnnWeight.h"

//MK: Batched Conv/ReLU/Pool/FC layers (CPU reference + GPU stubs + dispatch wrappers) - needs the shape macros from simpleCnnConfig.h and checkCudaError from cudaCommon.h
#include "layer.h"

//MK: argmaxLogits/imageToFloat - needs NUM_CLASSES (simpleCnnConfig.h) and CIFAR_IMAGE_SIZE (cifar10.h)
#include "../../common/simpleCnnUtil.h"

//MK: Main function
int main(){
	//MK: Load pre-trained weights
	float *h_conv1W = (float*)malloc(CONV1_W_SIZE * sizeof(float));
	float *h_conv1Bias = (float*)malloc(CONV1_OUT_C * sizeof(float));
	float *h_fc1W = (float*)malloc(FC1_W_SIZE * sizeof(float));
	float *h_fc1Bias = (float*)malloc(NUM_CLASSES * sizeof(float));

	if(!loadWeights(WEIGHTS_FILE, h_conv1W, h_conv1Bias, h_fc1W, h_fc1Bias)){
		printf("Error: could not open %s - the pre-trained weights file is missing from DB/.\n", WEIGHTS_FILE);
		return 1;
	}

	//MK: Set up whichever device buffers/weights the USE_GPU_* toggles in layer.h need
	initGpuBuffers();
	uploadWeights(h_conv1W, h_conv1Bias, h_fc1W, h_fc1Bias);

	//MK: Load CIFAR-10 test batch
	unsigned char *testLabels, *testImages;
	int numTest;
	cifar10Load(TEST_BATCH_FILE, &testLabels, &testImages, &numTest);
	int testCount = (TEST_IMAGES < numTest) ? TEST_IMAGES : numTest;

	//MK: Batch scratch buffers, sized for up to BATCH_SIZE images at once
	float *h_batchImages = (float*)malloc((size_t)BATCH_SIZE * CIFAR_IMAGE_SIZE * sizeof(float));
	float *h_batchConv1Out = (float*)malloc((size_t)BATCH_SIZE * CONV1_OUT_SIZE * sizeof(float));
	float *h_batchPoolOut = (float*)malloc((size_t)BATCH_SIZE * FLATTEN_SIZE * sizeof(float));
	int *h_batchPoolArgmax = (int*)malloc((size_t)BATCH_SIZE * FLATTEN_SIZE * sizeof(int));
	float *h_batchLogits = (float*)malloc((size_t)BATCH_SIZE * NUM_CLASSES * sizeof(float));

	clockMeasure *ckInfer = new clockMeasure("INFER TOTAL");
	clockMeasure *ckConv = new clockMeasure("Conv1");
	clockMeasure *ckRelu = new clockMeasure("ReLU");
	clockMeasure *ckPool = new clockMeasure("Pool");
	clockMeasure *ckFc = new clockMeasure("FC1");
	ckInfer->clockReset();
	ckConv->clockReset();
	ckRelu->clockReset();
	ckPool->clockReset();
	ckFc->clockReset();

	//MK: Inference - BATCH_SIZE images per step, each layer timed separately so you can see the effect of moving just that layer to the GPU
	int correct = 0;
	for(int batchStart = 0; batchStart < testCount; batchStart += BATCH_SIZE){
		int batchEnd = (batchStart + BATCH_SIZE < testCount) ? batchStart + BATCH_SIZE : testCount;
		int batchSize = batchEnd - batchStart;

		for(int i = 0; i < batchSize; i++){
			imageToFloat(testImages + (size_t)(batchStart + i) * CIFAR_IMAGE_SIZE, h_batchImages + (size_t)i * CIFAR_IMAGE_SIZE);
		}

		ckInfer->clockResume();

		ckConv->clockResume();
		runConv1(h_batchImages, h_conv1W, h_conv1Bias, h_batchConv1Out, batchSize);
		ckConv->clockPause();

		ckRelu->clockResume();
		runRelu(h_batchConv1Out, batchSize * CONV1_OUT_SIZE);
		ckRelu->clockPause();

		ckPool->clockResume();
		runPool(h_batchConv1Out, h_batchPoolOut, h_batchPoolArgmax, batchSize);
		ckPool->clockPause();

		ckFc->clockResume();
		runFc(h_batchPoolOut, h_fc1W, h_fc1Bias, h_batchLogits, batchSize);
		ckFc->clockPause();

		ckInfer->clockPause();

		for(int i = 0; i < batchSize; i++){
			if(argmaxLogits(h_batchLogits + (size_t)i * NUM_CLASSES) == testLabels[batchStart + i]) correct++;
		}
	}

	printf("\n=== Inference ===\n");
	printf("Top-1 Test Accuracy: %d/%d (%.2f%%)\n", correct, testCount, 100.0f * correct / testCount);
	verifyAccuracy(correct, testCount, EXPECTED_RESULT_FILE);
	ckInfer->clockPrint();
	printf("\t- "); ckConv->clockPrint();
	printf("\t- "); ckRelu->clockPrint();
	printf("\t- "); ckPool->clockPrint();
	printf("\t- "); ckFc->clockPrint();

	//MK: Cleanup
	free(testLabels); free(testImages);
	free(h_conv1W); free(h_conv1Bias); free(h_fc1W); free(h_fc1Bias);
	free(h_batchImages); free(h_batchConv1Out); free(h_batchPoolOut); free(h_batchPoolArgmax); free(h_batchLogits);
	freeGpuBuffers();

	return 0;
}
