CC=nvcc

simpleCNNBatchTilingGPUEmpty: main.cu layer.h ../../common/simpleCnnConfig.h ../../common/simpleCnnWeight.h ../../common/simpleCnnUtil.h ../../common/clockMeasure.h ../../common/cudaCommon.h ../../common/cifar10.h
	$(CC) -o simpleCNNBatchTilingGPUEmpty main.cu

clean:
	rm -rf simpleCNNBatchTilingGPUEmpty
