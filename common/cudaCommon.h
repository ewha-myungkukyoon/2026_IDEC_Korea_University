/*
 * =====================================================================================
 *
 *       Filename:  cudaCommon.h
 *
 *    Description:  Shared CUDA utilities used across example codes.
 *                  Currently provides the checkCudaError macro, which
 *                  checks the return value of CUDA runtime API calls,
 *                  prints the CUDA error string together with the
 *                  file/line where it occurred, and aborts the program.
 *
 *        Version:  1.0
 *        Created:  07/14/2026 01:20:00 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define checkCudaError(error)                                                \
	if (error != cudaSuccess) {                                              \
		printf("%s in %s at line %d\n", cudaGetErrorString(error),          \
		       __FILE__, __LINE__);                                          \
		exit(EXIT_FAILURE);                                                  \
	}
