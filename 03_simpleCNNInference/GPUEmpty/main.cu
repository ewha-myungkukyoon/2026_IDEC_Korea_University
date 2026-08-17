/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Exercise: port the simple CIFAR-10 CNN's inference to
 *                  the GPU one layer at a time. Conv1 -> ReLU -> MaxPool
 *                  -> FC1 on pre-trained weights, reporting Top-1 accuracy
 *                  and per-layer timing on the CIFAR-10 test set. Work in
 *                  layer.h - this file needs no changes.
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

#include <stdio.h>
#include <stdlib.h>
#include "../../common/clockMeasure.h"
#include "../../common/cifar10.h"
#include "../../common/cudaCommon.h"

//MK: All #define constants (network shapes + training config) live in common/simpleCnnConfig.h, shared with the GPU variant
#include "../../common/simpleCnnConfig.h"

//MK: loadWeights - needs the shape macros from simpleCnnConfig.h
#include "../../common/simpleCnnWeight.h"

//MK: Conv/ReLU/Pool/FC layers (CPU reference + GPU stubs + dispatch wrappers) - needs the shape macros from simpleCnnConfig.h and checkCudaError from cudaCommon.h
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

	//MK: Per-image scratch buffers
	float *h_inputImage = (float*)malloc(CIFAR_IMAGE_SIZE * sizeof(float));
	float *h_conv1Out = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));
	float *h_poolOut = (float*)malloc(FLATTEN_SIZE * sizeof(float));
	int *h_poolArgmax = (int*)malloc(FLATTEN_SIZE * sizeof(int));
	float h_logits[NUM_CLASSES];

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

	//MK: Inference - each layer timed separately so you can see the effect of moving just that layer to the GPU
	int correct = 0;
	for(int i = 0; i < testCount; i++){
		imageToFloat(testImages + (size_t)i * CIFAR_IMAGE_SIZE, h_inputImage);
		int label = testLabels[i];

		ckInfer->clockResume();

		ckConv->clockResume();
		runConv1(h_inputImage, h_conv1W, h_conv1Bias, h_conv1Out);
		ckConv->clockPause();

		ckRelu->clockResume();
		runRelu(h_conv1Out, CONV1_OUT_SIZE);
		ckRelu->clockPause();

		ckPool->clockResume();
		runPool(h_conv1Out, h_poolOut, h_poolArgmax);
		ckPool->clockPause();

		ckFc->clockResume();
		runFc(h_poolOut, h_fc1W, h_fc1Bias, h_logits);
		ckFc->clockPause();

		ckInfer->clockPause();

		if(argmaxLogits(h_logits) == label) correct++;
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
	free(h_inputImage); free(h_conv1Out); free(h_poolOut); free(h_poolArgmax);
	freeGpuBuffers();

	return 0;
}
