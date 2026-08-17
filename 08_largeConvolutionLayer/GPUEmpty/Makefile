CC=nvcc

#MK: -O3 matters here specifically because main.cu times a scalar CPU direct-conv reference over a 3.7 GFLOP
#MK: workload 10x - without it nvcc's default (unoptimized) host codegen makes the CPU baseline misleadingly slow.
#MK: -arch=sm_86 matters because main.cu's gemmTensorCore kernel uses the WMMA Tensor Core API (needs sm_70+);
#MK: nvcc's default target architecture is too old for it to compile. sm_86 = Ampere (RTX 3090, CC 8.6) -
#MK: adjust if building for a different GPU generation.
largeConvolutionLayerGPUEmpty: main.cu ../../common/clockMeasure.h ../../common/cudaCommon.h
	$(CC) -O3 -arch=sm_86 -o largeConvolutionLayerGPUEmpty main.cu

clean:
	rm -rf largeConvolutionLayerGPUEmpty
