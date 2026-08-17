/*
 * =====================================================================================
 *
 *       Filename:  simpleCnnVerify.h
 *
 *    Description:  CPU-vs-GPU verification for the simple CIFAR-10 CNN
 *                  training example, in two layers:
 *
 *                  1) verifyGpuKernels() - a per-kernel check run once
 *                     before training starts. Every GPU kernel is fed the
 *                     CPU reference implementation's inputs and its output
 *                     is compared against the CPU reference's output, so a
 *                     FAIL points at that one kernel instead of at whatever
 *                     ran before it. This is the harness to lean on while
 *                     filling in the GPUEmpty exercise.
 *
 *                  2) compareTrainedModels() - run after both models have
 *                     trained, to answer "did the CPU and the GPU learn the
 *                     same thing?".
 *
 *                  Neither check tests for equality. float32 arithmetic
 *                  cannot match bit-for-bit across the two paths: the GPU
 *                  contracts a*b+c into a single FMA (one rounding instead
 *                  of two) while the host code does not, so identical math
 *                  in identical order still lands on slightly different
 *                  bits. So every comparison is bounded instead:
 *
 *                    per-element:  |cpu - gpu| <= VERIFY_ABS_TOL + VERIFY_REL_TOL * |cpu|
 *
 *                  The absolute term is what makes the bound usable near
 *                  zero (a relative bound alone rejects 1e-9 vs 2e-9), and
 *                  the relative term is what makes it usable on large
 *                  values (an absolute bound alone rejects 1000.0 vs
 *                  1000.001). A single kernel's output should sit orders of
 *                  magnitude inside this bound - the defaults are loose
 *                  enough that only a real bug trips them.
 *
 *                  compareTrainedModels() cannot use that bound. Two models
 *                  that trained independently for EPOCHS * TRAIN_IMAGES
 *                  steps amplify a 1e-7 rounding difference into a visibly
 *                  different (but equally good) model, so it reports the
 *                  relative L2 distance between the two weight vectors plus
 *                  the gap in final test accuracy, and bounds those.
 *
 *                  Expects the network-shape macros and the VERIFY_*
 *                  configuration (simpleCnnConfig.h), CIFAR_IMAGE_SIZE
 *                  (cifar10.h), checkCudaError (cudaCommon.h), the layer
 *                  kernel/CPU function pairs plus THREADS and numBlocks()
 *                  (layer.h), imageToFloat (simpleCnnUtil.h), and
 *                  softmaxCrossEntropy (defined in main.cu) to already be
 *                  available - include this header last, see main.cu.
 *
 *        Version:  1.0
 *        Created:  08/15/2026 11:00:00 AM
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
#include <stdlib.h>
#include <string.h>
#include <math.h>

//MK: ==================== Comparison primitives ====================

//MK: True only for a finite value - written as a negated <= so that NaN (which fails every comparison) lands here too
bool isFiniteFloat(float v){
	return (fabsf(v) <= 3.402823466e38f);
}

//MK: Element-wise bounded comparison: |ref - test| <= absTol + relTol * |ref|. Prints one summary line and
//MK: returns whether every element stayed inside the bound.
bool compareArray(const char *name, const float *ref, const float *test, int size, float relTol, float absTol){
	float maxAbsDiff = 0.0f, worstRatio = 0.0f;
	int worstIdx = 0, badCount = 0, notFiniteCount = 0;

	for(int i = 0; i < size; i++){
		float diff = fabsf(ref[i] - test[i]);
		float tol = absTol + relTol * fabsf(ref[i]);

		//MK: Written as !(diff <= tol) rather than (diff > tol) so a NaN difference counts as a failure
		if(!(diff <= tol)) badCount++;
		if(!isFiniteFloat(test[i])) notFiniteCount++;

		//MK: diff/tol is the one number worth reporting: it says how much of the allowance this element used,
		//MK: so 1.0 is exactly the pass/fail line no matter which of the two terms dominates the bound here.
		//MK: (A bare relative difference would look alarming on values that are ~0, where the absolute term is
		//MK: doing all the work and the element is nowhere near failing.)
		float ratio = diff / tol;
		if(ratio > worstRatio) worstRatio = ratio;

		if(diff > maxAbsDiff){
			maxAbsDiff = diff;
			worstIdx = i;
		}
	}

	bool pass = (badCount == 0);
	printf("  %-21s %s  max|diff| %.3e  worst diff/tol %.4f  outside bound %d/%d\n",
			name, pass ? "PASS" : "FAIL", maxAbsDiff, worstRatio, badCount, size);
	if(!pass){
		printf("      worst element [%d]: cpu %.8e vs gpu %.8e\n", worstIdx, ref[worstIdx], test[worstIdx]);
		if(notFiniteCount > 0){
			printf("      %d element(s) are NaN/Inf - the kernel is most likely not writing every output element\n", notFiniteCount);
		}
	}
	return pass;
}

//MK: The pooling argmax is an index, not a measurement - fed identical inputs both sides must pick the identical winner
bool compareIntArray(const char *name, const int *ref, const int *test, int size){
	int badCount = 0, worstIdx = -1;
	for(int i = 0; i < size; i++){
		if(ref[i] != test[i]){
			badCount++;
			if(worstIdx < 0) worstIdx = i;
		}
	}

	bool pass = (badCount == 0);
	printf("  %-21s %s  exact match required          mismatched %d/%d\n", name, pass ? "PASS" : "FAIL", badCount, size);
	if(!pass) printf("      first mismatch [%d]: cpu %d vs gpu %d\n", worstIdx, ref[worstIdx], test[worstIdx]);
	return pass;
}

//MK: ||ref - test||2 / ||ref||2 - accumulated in double so the metric itself doesn't add float error
float relL2Diff(const float *ref, const float *test, int size){
	double num = 0.0, den = 0.0;
	for(int i = 0; i < size; i++){
		double d = (double)ref[i] - (double)test[i];
		num += d * d;
		den += (double)ref[i] * (double)ref[i];
	}
	if(den == 0.0) return (num == 0.0) ? 0.0f : 1.0f;
	return (float)sqrt(num / den);
}

//MK: ==================== Per-kernel verification ====================

//MK: Runs the CPU reference forward+backward on a few images, then replays every GPU kernel with the CPU
//MK: reference's own inputs uploaded into the device buffers first. Feeding known-good inputs is what keeps
//MK: one broken kernel from cascading into a wall of failures downstream. Weights are never updated here.
bool verifyGpuKernels(const float *h_conv1W, const float *h_conv1Bias, const float *h_fc1W, const float *h_fc1Bias,
		const unsigned char *images, const unsigned char *labels, int numImages){
	int verifyCount = (VERIFY_IMAGES < numImages) ? VERIFY_IMAGES : numImages;
	if(verifyCount < 1) verifyCount = 1;

	printf("\n=== Kernel Verification: CPU reference vs GPU (%d image(s)) ===\n", verifyCount);
	printf("Bound: |cpu - gpu| <= %.1e + %.1e * |cpu|   (float math is not bit-exact - the GPU folds a*b+c into one FMA)\n",
			VERIFY_ABS_TOL, VERIFY_REL_TOL);
	printf("Each kernel is fed the CPU reference's inputs, so a FAIL names the one kernel that is wrong.\n");

	cudaError_t err;
	int failCount = 0;

	//MK: Host reference buffers - one per intermediate tensor, since the in-place layers (ReLU) would otherwise
	//MK: destroy the very value the next comparison needs
	float *h_input = (float*)malloc(CIFAR_IMAGE_SIZE * sizeof(float));
	float *refConv = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));
	float *refRelu = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));
	float *refPool = (float*)malloc(FLATTEN_SIZE * sizeof(float));
	int *refArgmax = (int*)malloc(FLATTEN_SIZE * sizeof(int));
	float *refLogits = (float*)malloc(NUM_CLASSES * sizeof(float));
	float *refDLogits = (float*)malloc(NUM_CLASSES * sizeof(float));
	float *refFc1dW = (float*)malloc(FC1_W_SIZE * sizeof(float));
	float *refDPool = (float*)malloc(FLATTEN_SIZE * sizeof(float));
	float *refDConvPool = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));
	float *refDConvRelu = (float*)malloc(CONV1_OUT_SIZE * sizeof(float));
	float *refConv1dW = (float*)malloc(CONV1_W_SIZE * sizeof(float));
	float *refConv1dBias = (float*)malloc(CONV1_OUT_C * sizeof(float));
	float *refScratch = (float*)malloc(FC1_W_SIZE * sizeof(float));

	//MK: Landing buffer for whatever the GPU just produced - sized for the largest tensor checked
	float *gpuOut = (float*)malloc(FC1_W_SIZE * sizeof(float));
	int *gpuOutInt = (int*)malloc(FLATTEN_SIZE * sizeof(int));

	//MK: This check owns its own device buffers so it stays independent of however main() lays out its own
	float *v_input, *v_convOut, *v_poolOut, *v_logits;
	float *v_dLogits, *v_dPoolOut, *v_dConvOut;
	float *v_conv1W, *v_conv1Bias, *v_fc1W, *v_fc1Bias;
	float *v_conv1dW, *v_conv1dBias, *v_fc1dW, *v_scratch;
	int *v_poolArgmax;

	err = cudaMalloc((void**)&v_input, CIFAR_IMAGE_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_convOut, CONV1_OUT_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_poolOut, FLATTEN_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_poolArgmax, FLATTEN_SIZE * sizeof(int)); checkCudaError(err);
	err = cudaMalloc((void**)&v_logits, NUM_CLASSES * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_dLogits, NUM_CLASSES * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_dPoolOut, FLATTEN_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_dConvOut, CONV1_OUT_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_conv1W, CONV1_W_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_conv1Bias, CONV1_OUT_C * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_fc1W, FC1_W_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_fc1Bias, NUM_CLASSES * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_conv1dW, CONV1_W_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_conv1dBias, CONV1_OUT_C * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_fc1dW, FC1_W_SIZE * sizeof(float)); checkCudaError(err);
	err = cudaMalloc((void**)&v_scratch, FC1_W_SIZE * sizeof(float)); checkCudaError(err);

	err = cudaMemcpy(v_conv1W, h_conv1W, CONV1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
	err = cudaMemcpy(v_conv1Bias, h_conv1Bias, CONV1_OUT_C * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
	err = cudaMemcpy(v_fc1W, h_fc1W, FC1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
	err = cudaMemcpy(v_fc1Bias, h_fc1Bias, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);

	for(int n = 0; n < verifyCount; n++){
		imageToFloat(images + (size_t)n * CIFAR_IMAGE_SIZE, h_input);
		int label = labels[n];
		printf("-- Image %d (label %d) --\n", n, label);

		//MK: ---------- CPU reference: one full forward + backward, every intermediate kept ----------
		conv1ForwardCpu(h_input, h_conv1W, h_conv1Bias, refConv);
		memcpy(refRelu, refConv, CONV1_OUT_SIZE * sizeof(float));
		reluForwardCpu(refRelu, CONV1_OUT_SIZE);
		poolForwardCpu(refRelu, refPool, refArgmax);
		fcForwardCpu(refPool, h_fc1W, h_fc1Bias, refLogits);
		softmaxCrossEntropy(refLogits, label, refDLogits);

		fcBackwardWeightCpu(refDLogits, refPool, refFc1dW);
		fcBackwardInputCpu(refDLogits, h_fc1W, refDPool);
		memset(refDConvPool, 0, CONV1_OUT_SIZE * sizeof(float));
		poolBackwardCpu(refDPool, refArgmax, refDConvPool);
		memcpy(refDConvRelu, refDConvPool, CONV1_OUT_SIZE * sizeof(float));
		reluBackwardCpu(refRelu, refDConvRelu, CONV1_OUT_SIZE);
		conv1BackwardWeightCpu(h_input, refDConvRelu, refConv1dW);
		conv1BackwardBiasCpu(refDConvRelu, refConv1dBias);

		//MK: ---------- GPU: one kernel at a time, each with the reference's inputs uploaded first ----------

		//MK: conv1Forward - input image in, conv output out
		err = cudaMemcpy(v_input, h_input, CIFAR_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		conv1Forward<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(v_input, v_conv1W, v_conv1Bias, v_convOut);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_convOut, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("conv1Forward", refConv, gpuOut, CONV1_OUT_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

		//MK: reluForward - in-place, so upload the pre-ReLU reference every time
		err = cudaMemcpy(v_convOut, refConv, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		reluForward<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(v_convOut, CONV1_OUT_SIZE);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_convOut, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("reluForward", refRelu, gpuOut, CONV1_OUT_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

		//MK: poolForward - two outputs to check, the pooled values and the argmax indices
		err = cudaMemcpy(v_convOut, refRelu, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		poolForward<<<numBlocks(FLATTEN_SIZE), THREADS>>>(v_convOut, v_poolOut, v_poolArgmax);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_poolOut, FLATTEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		err = cudaMemcpy(gpuOutInt, v_poolArgmax, FLATTEN_SIZE * sizeof(int), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("poolForward", refPool, gpuOut, FLATTEN_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;
		if(!compareIntArray("poolForward(argmax)", refArgmax, gpuOutInt, FLATTEN_SIZE)) failCount++;

		//MK: fcForward
		err = cudaMemcpy(v_poolOut, refPool, FLATTEN_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		fcForward<<<numBlocks(NUM_CLASSES), THREADS>>>(v_poolOut, v_fc1W, v_fc1Bias, v_logits);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_logits, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("fcForward", refLogits, gpuOut, NUM_CLASSES, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

		//MK: fcBackwardWeight - dLogits comes from the CPU softmax on both paths, exactly as in training
		err = cudaMemcpy(v_dLogits, refDLogits, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		fcBackwardWeight<<<numBlocks(FC1_W_SIZE), THREADS>>>(v_dLogits, v_poolOut, v_fc1dW);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_fc1dW, FC1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("fcBackwardWeight", refFc1dW, gpuOut, FC1_W_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

		//MK: fcBackwardInput
		fcBackwardInput<<<numBlocks(FLATTEN_SIZE), THREADS>>>(v_dLogits, v_fc1W, v_dPoolOut);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_dPoolOut, FLATTEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("fcBackwardInput", refDPool, gpuOut, FLATTEN_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

		//MK: poolBackward - the memset is part of the step being verified, not a setup detail (scatter only touches the winners)
		err = cudaMemcpy(v_dPoolOut, refDPool, FLATTEN_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		err = cudaMemcpy(v_poolArgmax, refArgmax, FLATTEN_SIZE * sizeof(int), cudaMemcpyHostToDevice); checkCudaError(err);
		err = cudaMemset(v_dConvOut, 0, CONV1_OUT_SIZE * sizeof(float)); checkCudaError(err);
		poolBackward<<<numBlocks(FLATTEN_SIZE), THREADS>>>(v_dPoolOut, v_poolArgmax, v_dConvOut);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_dConvOut, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("poolBackward", refDConvPool, gpuOut, CONV1_OUT_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

		//MK: reluBackward - needs the ReLU *output* as the mask and the pool-backward result as the gradient
		err = cudaMemcpy(v_convOut, refRelu, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		err = cudaMemcpy(v_dConvOut, refDConvPool, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		reluBackward<<<numBlocks(CONV1_OUT_SIZE), THREADS>>>(v_convOut, v_dConvOut, CONV1_OUT_SIZE);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_dConvOut, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("reluBackward", refDConvRelu, gpuOut, CONV1_OUT_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

		//MK: conv1BackwardWeight
		err = cudaMemcpy(v_dConvOut, refDConvRelu, CONV1_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
		conv1BackwardWeight<<<numBlocks(CONV1_W_SIZE), THREADS>>>(v_input, v_dConvOut, v_conv1dW);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_conv1dW, CONV1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("conv1BackwardWeight", refConv1dW, gpuOut, CONV1_W_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

		//MK: conv1BackwardBias
		conv1BackwardBias<<<numBlocks(CONV1_OUT_C), THREADS>>>(v_dConvOut, v_conv1dBias);
		err = cudaDeviceSynchronize(); checkCudaError(err);
		err = cudaMemcpy(gpuOut, v_conv1dBias, CONV1_OUT_C * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
		if(!compareArray("conv1BackwardBias", refConv1dBias, gpuOut, CONV1_OUT_C, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;
	}

	//MK: ---------- The two image-independent kernels, checked once on FC1-sized data ----------
	printf("-- Optimizer kernels --\n");

	//MK: accumulateGrad - add the same gradient twice, so the reference is exactly 2x the input
	memset(refScratch, 0, FC1_W_SIZE * sizeof(float));
	accumulateGradCpu(refScratch, refFc1dW, FC1_W_SIZE);
	accumulateGradCpu(refScratch, refFc1dW, FC1_W_SIZE);
	err = cudaMemcpy(v_fc1dW, refFc1dW, FC1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
	err = cudaMemset(v_scratch, 0, FC1_W_SIZE * sizeof(float)); checkCudaError(err);
	accumulateGrad<<<numBlocks(FC1_W_SIZE), THREADS>>>(v_scratch, v_fc1dW, FC1_W_SIZE);
	accumulateGrad<<<numBlocks(FC1_W_SIZE), THREADS>>>(v_scratch, v_fc1dW, FC1_W_SIZE);
	err = cudaDeviceSynchronize(); checkCudaError(err);
	err = cudaMemcpy(gpuOut, v_scratch, FC1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
	if(!compareArray("accumulateGrad", refScratch, gpuOut, FC1_W_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

	//MK: sgdUpdate - one step on the real fc1 weights with the real batch learning rate
	float verifyLR = LEARNING_RATE / BATCH_SIZE;
	memcpy(refScratch, h_fc1W, FC1_W_SIZE * sizeof(float));
	sgdUpdateCpu(refScratch, refFc1dW, verifyLR, FC1_W_SIZE);
	err = cudaMemcpy(v_scratch, h_fc1W, FC1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice); checkCudaError(err);
	sgdUpdate<<<numBlocks(FC1_W_SIZE), THREADS>>>(v_scratch, v_fc1dW, verifyLR, FC1_W_SIZE);
	err = cudaDeviceSynchronize(); checkCudaError(err);
	err = cudaMemcpy(gpuOut, v_scratch, FC1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost); checkCudaError(err);
	if(!compareArray("sgdUpdate", refScratch, gpuOut, FC1_W_SIZE, VERIFY_REL_TOL, VERIFY_ABS_TOL)) failCount++;

	free(h_input); free(refConv); free(refRelu); free(refPool); free(refArgmax);
	free(refLogits); free(refDLogits); free(refFc1dW); free(refDPool);
	free(refDConvPool); free(refDConvRelu); free(refConv1dW); free(refConv1dBias);
	free(refScratch); free(gpuOut); free(gpuOutInt);

	cudaFree(v_input); cudaFree(v_convOut); cudaFree(v_poolOut); cudaFree(v_poolArgmax); cudaFree(v_logits);
	cudaFree(v_dLogits); cudaFree(v_dPoolOut); cudaFree(v_dConvOut);
	cudaFree(v_conv1W); cudaFree(v_conv1Bias); cudaFree(v_fc1W); cudaFree(v_fc1Bias);
	cudaFree(v_conv1dW); cudaFree(v_conv1dBias); cudaFree(v_fc1dW); cudaFree(v_scratch);

	if(failCount == 0){
		printf("=== Kernel Verification PASSED - every GPU kernel matches its CPU counterpart ===\n");
	} else {
		printf("=== Kernel Verification FAILED - %d check(s) outside the bound (see the FAIL lines above) ===\n", failCount);
	}
	return failCount == 0;
}

//MK: ==================== Post-training model comparison ====================

//MK: One weight tensor's worth of "did these two end up in the same place?" - reported, then bounded by relative L2
bool compareWeightTensor(const char *name, const float *cpu, const float *gpu, int size){
	float maxAbsDiff = 0.0f;
	for(int i = 0; i < size; i++){
		float d = fabsf(cpu[i] - gpu[i]);
		if(d > maxAbsDiff) maxAbsDiff = d;
	}
	float l2 = relL2Diff(cpu, gpu, size);
	bool pass = (l2 <= VERIFY_WEIGHT_L2_TOL);
	printf("  %-14s %s  relative L2 %.4f (bound %.4f)  max|diff| %.3e  (%d weights)\n",
			name, pass ? "PASS" : "FAIL", l2, VERIFY_WEIGHT_L2_TOL, maxAbsDiff, size);
	return pass;
}

//MK: Answers "did the CPU model and the GPU model learn the same thing?" after both have trained from the same
//MK: initial weights on the same image order. Element-wise equality is the wrong question here: the two runs
//MK: differ by ~1e-7 on the very first image, and EPOCHS * TRAIN_IMAGES SGD steps amplify that - a single
//MK: flipped ReLU sign or pooling winner sends the two models down slightly different paths. What has to hold
//MK: is that they landed on the same solution: weight vectors pointing the same way (relative L2) and the same
//MK: test accuracy.
bool compareTrainedModels(const float *conv1W_cpu, const float *conv1Bias_cpu, const float *fc1W_cpu, const float *fc1Bias_cpu,
		const float *conv1W_gpu, const float *conv1Bias_gpu, const float *fc1W_gpu, const float *fc1Bias_gpu,
		int correctCpu, int correctGpu, int testCount){
	printf("\n=== Trained Model Comparison: CPU vs GPU ===\n");
	printf("Both models started from identical weights and saw identical images in identical order, but float32\n");
	printf("rounding differences compound over %d SGD steps - so this bounds how far apart they ended up, not\n",
			EPOCHS * (TRAIN_IMAGES / BATCH_SIZE));
	printf("whether they are equal.\n");

	int failCount = 0;
	if(!compareWeightTensor("conv1 weight", conv1W_cpu, conv1W_gpu, CONV1_W_SIZE)) failCount++;
	if(!compareWeightTensor("conv1 bias", conv1Bias_cpu, conv1Bias_gpu, CONV1_OUT_C)) failCount++;
	if(!compareWeightTensor("fc1 weight", fc1W_cpu, fc1W_gpu, FC1_W_SIZE)) failCount++;
	if(!compareWeightTensor("fc1 bias", fc1Bias_cpu, fc1Bias_gpu, NUM_CLASSES)) failCount++;

	//MK: The one number that actually matters - two models can differ in weights and still be equally good
	float accCpu = 100.0f * correctCpu / testCount;
	float accGpu = 100.0f * correctGpu / testCount;
	float accGap = fabsf(accCpu - accGpu);
	bool accPass = (accGap <= VERIFY_ACC_TOL);
	printf("  %-14s %s  CPU %.2f%% vs GPU %.2f%%  gap %.2f%%p (bound %.2f%%p)\n",
			"test accuracy", accPass ? "PASS" : "FAIL", accCpu, accGpu, accGap, VERIFY_ACC_TOL);
	if(!accPass) failCount++;

	if(failCount == 0){
		printf("=== Trained Model Comparison PASSED - the CPU and GPU models learned the same thing ===\n");
	} else {
		printf("=== Trained Model Comparison FAILED - %d check(s) outside the bound ===\n", failCount);
		printf("(If the per-kernel verification passed, suspect the training loop wiring - a missing memset of a\n");
		printf(" gradient accumulator, a wrong learning rate, or a kernel launched with the wrong element count.)\n");
	}
	return failCount == 0;
}
