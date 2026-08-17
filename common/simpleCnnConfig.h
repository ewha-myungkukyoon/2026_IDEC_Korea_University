/*
 * =====================================================================================
 *
 *       Filename:  simpleCnnConfig.h
 *
 *    Description:  All #define constants for the simple CIFAR-10 CNN -
 *                  network shapes (fixed to the reference architecture
 *                  table) and data/training configuration. Shared by
 *                  every simpleCNN variant folder (CPU and GPU alike)
 *                  so they all agree on the same network shape and file
 *                  paths. The path defines (TRAIN_BATCH_FILE_1..5,
 *                  TEST_BATCH_FILE, WEIGHTS_FILE) are resolved relative
 *                  to the working directory the built binary is run
 *                  from, not relative to this header - every variant
 *                  folder sits at the same depth under cuda/, so this
 *                  works out unchanged.
 *
 *        Version:  1.0
 *        Created:  07/14/2026 04:20:00 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

#pragma once

//MK: Network shapes (fixed to the reference architecture table)
#define IN_C 3
#define IN_H 32
#define IN_W 32

#define CONV1_OUT_C 16
#define CONV1_K 3
#define CONV1_OUT_H (IN_H - CONV1_K + 1)   //30
#define CONV1_OUT_W (IN_W - CONV1_K + 1)   //30
#define CONV1_OUT_SIZE (CONV1_OUT_C * CONV1_OUT_H * CONV1_OUT_W)
#define CONV1_W_SIZE (CONV1_OUT_C * IN_C * CONV1_K * CONV1_K)  //432

#define POOL1_K 2
#define POOL1_OUT_H (CONV1_OUT_H / POOL1_K) //15
#define POOL1_OUT_W (CONV1_OUT_W / POOL1_K) //15
#define FLATTEN_SIZE (CONV1_OUT_C * POOL1_OUT_H * POOL1_OUT_W) //3600

#define NUM_CLASSES 10
#define FC1_W_SIZE (NUM_CLASSES * FLATTEN_SIZE) //36000

//MK: Data/training configuration - all 5 training batches (50000 images total)
#define TRAIN_BATCH_FILE_1 "../../DB/cifar-10-batches-bin/data_batch_1.bin"
#define TRAIN_BATCH_FILE_2 "../../DB/cifar-10-batches-bin/data_batch_2.bin"
#define TRAIN_BATCH_FILE_3 "../../DB/cifar-10-batches-bin/data_batch_3.bin"
#define TRAIN_BATCH_FILE_4 "../../DB/cifar-10-batches-bin/data_batch_4.bin"
#define TRAIN_BATCH_FILE_5 "../../DB/cifar-10-batches-bin/data_batch_5.bin"
#define TEST_BATCH_FILE  "../../DB/cifar-10-batches-bin/test_batch.bin"
#define WEIGHTS_FILE     "../../DB/03_simpleCNN-weight/weights.bin"
#define EXPECTED_RESULT_FILE "../../DB/03_simpleCNN-weight/expected_result.txt"

#define TRAIN_IMAGES 50000
#define TEST_IMAGES  10000
#define EPOCHS 5
#define BATCH_SIZE 64
#define LEARNING_RATE 0.01f
#define PRINT_EVERY 10000
#define WEIGHT_INIT_SCALE 0.05f

//MK: There is deliberately no SAVE_WEIGHTS switch here and no writer in simpleCnnWeight.h: nothing in these
//MK: examples may overwrite the pre-trained DB/03_simpleCNN-weight/weights.bin + expected_result.txt that
//MK: 03 through 08 load and verify against. 09 trains from scratch to show the loss come down, then exits
//MK: without touching DB/ - its run is not reproducible anyway (weights are seeded from srand(time(NULL))),
//MK: and it is sensitive to the settings above: BATCH_SIZE 64 with EPOCHS 5 reaches ~44% Top-1 while
//MK: BATCH_SIZE 32 reaches ~52%, since the same LEARNING_RATE/batchSize step taken half as many times is
//MK: simply half the training.

//MK: Verification configuration (see simpleCnnVerify.h) - every GPU kernel is checked against its CPU
//MK: counterpart before training starts. float32 results can't match bit-for-bit across the two paths (the
//MK: GPU contracts a*b+c into a single FMA, one rounding instead of two), so the check is bounded instead of
//MK: exact: |cpu - gpu| <= VERIFY_ABS_TOL + VERIFY_REL_TOL * |cpu|. The absolute term keeps the bound usable
//MK: near zero, the relative term keeps it usable on large values. A correct kernel lands orders of magnitude
//MK: inside these defaults, so only a real bug trips them.
#define VERIFY_KERNELS 1        //MK: 0 to skip the per-kernel check
#define VERIFY_IMAGES 3         //MK: how many images to push through it
#define VERIFY_STOP_ON_FAIL 1   //MK: 1 = quit instead of starting a long training run with a broken kernel
#define VERIFY_REL_TOL 1e-4f
#define VERIFY_ABS_TOL 1e-5f

//MK: Post-training CPU-vs-GPU model comparison (needs RUN_BOTH). Two models that trained independently in
//MK: float32 amplify that same 1e-7 difference over thousands of SGD steps, so the per-element bound above is
//MK: the wrong tool here - this one bounds the relative L2 distance between the two weight vectors and the gap
//MK: in final test accuracy, which is what "they learned the same thing" actually means.
#define VERIFY_TRAINED_MODEL 1
#define VERIFY_WEIGHT_L2_TOL 0.05f   //MK: 5% relative L2 distance between the CPU and GPU weight vectors
#define VERIFY_ACC_TOL 1.0f          //MK: allowed Top-1 accuracy gap, in percentage points

//MK: Which implementation(s) to run - RUN_CPU_ONLY skips every CUDA call, so it can run without a working GPU
enum RunMode { RUN_CPU_ONLY = 0, RUN_GPU_ONLY = 1, RUN_BOTH = 2 };
#define RUN_MODE RUN_BOTH
