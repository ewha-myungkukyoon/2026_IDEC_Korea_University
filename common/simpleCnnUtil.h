/*
 * =====================================================================================
 *
 *       Filename:  simpleCnnUtil.h
 *
 *    Description:  Small shared helpers for the simple CIFAR-10 CNN,
 *                  used by every CPU/GPU variant: converting a raw
 *                  CIFAR-10 image to a normalized float buffer, reading
 *                  off the predicted class from the 10 logits, and
 *                  verifying the final Top-1 count against the expected
 *                  result shipped alongside the pre-trained weights (see
 *                  simpleCnnWeight.h's loadExpectedResult - the expected
 *                  count lives in a file next to weights.bin, not a
 *                  hardcoded constant here, so the two always describe
 *                  the same model).
 *
 *                  argmaxLogits expects NUM_CLASSES (simpleCnnConfig.h),
 *                  verifyAccuracy expects loadExpectedResult to already
 *                  be declared (simpleCnnWeight.h, included before this
 *                  header), and imageToFloat expects CIFAR_IMAGE_SIZE
 *                  (cifar10.h) - include all of those first.
 *
 *        Version:  1.0
 *        Created:  07/16/2026 02:00:00 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

#pragma once

//MK: argmax over the 10 logits (prediction doesn't need softmax, just the largest logit)
int argmaxLogits(const float *logits){
	int best = 0;
	for(int i = 1; i < NUM_CLASSES; i++){
		if(logits[i] > logits[best]) best = i;
	}
	return best;
}

//MK: Converts one CIFAR-10 image (unsigned char, CHW) to a normalized float buffer
void imageToFloat(const unsigned char *src, float *dst){
	for(int i = 0; i < CIFAR_IMAGE_SIZE; i++){
		dst[i] = src[i] / 255.0f;
	}
}

//MK: Checks the Top-1 count against the expected result saved next to the weights (expectedResultFile) - a
//MK: mismatch almost always means the GPU kernel (or the CPU/GPU wiring) has a bug, since the same pretrained
//MK: weights make the result deterministic
void verifyAccuracy(int correct, int testCount, const char *expectedResultFile){
	int expectedCorrect, expectedTotal;
	if(!loadExpectedResult(expectedResultFile, &expectedCorrect, &expectedTotal)){
		printf("[VERIFY] Skipped - could not open %s (run the training program first)\n", expectedResultFile);
		return;
	}
	if(testCount != expectedTotal){
		printf("[VERIFY] Skipped - testCount (%d) != expected test set size (%d)\n", testCount, expectedTotal);
		return;
	}
	if(correct == expectedCorrect){
		printf("[VERIFY] PASS - %d/%d matches the expected result recorded in %s\n", correct, testCount, expectedResultFile);
	} else {
		printf("[VERIFY] FAIL - expected %d/%d correct but got %d/%d - check your GPU kernel implementation\n",
				expectedCorrect, expectedTotal, correct, testCount);
	}
}
