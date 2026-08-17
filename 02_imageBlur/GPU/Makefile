CC=nvcc

imageBlurGPU: main.cu ../../common/ppm.h ../../common/clockMeasure.h ../../common/cudaCommon.h
	$(CC) -o imageBlurGPU main.cu

clean:
	rm -rf imageBlurGPU *.ppm
