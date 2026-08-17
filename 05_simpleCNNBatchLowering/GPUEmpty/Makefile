CC=nvcc

simpleCNNBatchLoweringGPUEmpty: main.cu layer.h ../../common/simpleCnnConfig.h ../../common/simpleCnnWeight.h ../../common/simpleCnnUtil.h ../../common/clockMeasure.h ../../common/cudaCommon.h ../../common/cifar10.h
	$(CC) -o simpleCNNBatchLoweringGPUEmpty main.cu

clean:
	rm -rf simpleCNNBatchLoweringGPUEmpty
