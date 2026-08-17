/*
 * =====================================================================================
 *
 *       Filename:  simpleCnnWeight.h
 *
 *    Description:  Loaders for the simple CIFAR-10 CNN's conv1 + fc1
 *                  weights and biases, stored as one flat binary file,
 *                  plus the Top-1 result (correct/total) that weights
 *                  file is expected to produce. Both files ship
 *                  pre-trained in DB/03_simpleCNN-weight/ and every
 *                  example here only ever reads them (loadWeights +
 *                  loadExpectedResult, the latter via simpleCnnUtil.h's
 *                  verifyAccuracy), so the "known good" answer always
 *                  travels with the weights it came from instead of
 *                  being a separate hardcoded constant.
 *
 *                  There are deliberately no writers here. Retraining is
 *                  not reproducible (weights are seeded from
 *                  srand(time(NULL))), so a run that overwrote these two
 *                  files would silently invalidate the expected result
 *                  every inference example verifies against - and 09's
 *                  training exercise is meant to show the loss come down,
 *                  not to replace the reference model.
 *
 *                  Expects CONV1_W_SIZE, CONV1_OUT_C, FC1_W_SIZE, and
 *                  NUM_CLASSES to already be #defined by whoever
 *                  includes this header - see simpleCnnConfig.h, which
 *                  must be included first.
 *
 *        Version:  1.0
 *        Created:  07/16/2026 00:40:00 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

#pragma once
#include <stdio.h>

//MK: Loads the pre-trained conv1 + fc1 weights/biases; returns false if the file doesn't exist
bool loadWeights(const char *filename, float *conv1W, float *conv1Bias, float *fc1W, float *fc1Bias){
	FILE *fp = fopen(filename, "rb");
	if(!fp) return false;

	fread(conv1W, sizeof(float), CONV1_W_SIZE, fp);
	fread(conv1Bias, sizeof(float), CONV1_OUT_C, fp);
	fread(fc1W, sizeof(float), FC1_W_SIZE, fp);
	fread(fc1Bias, sizeof(float), NUM_CLASSES, fp);
	fclose(fp);
	printf("(Weights loaded from %s)\n", filename);
	return true;
}

//MK: Loads the Top-1 result the pre-trained weights are known to produce; returns false if the file doesn't exist
bool loadExpectedResult(const char *filename, int *correct, int *total){
	FILE *fp = fopen(filename, "r");
	if(!fp) return false;

	int n = fscanf(fp, "%d %d", correct, total);
	fclose(fp);
	return n == 2;
}
