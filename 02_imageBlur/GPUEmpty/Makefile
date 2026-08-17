CC=nvcc

imageBlurGPUEmpty: main.cu ../../common/ppm.h ../../common/clockMeasure.h ../../common/cudaCommon.h
	$(CC) -o imageBlurGPUEmpty main.cu

clean:
	rm -rf imageBlurGPUEmpty *.ppm
