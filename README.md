# 2026_IDEC_Korea_University

Lecture materials for the IDEC course at Korea University, August 24-26, 2026.

## Exercises

| Directory | Description |
| --- | --- |
| `01_vectorAdd` | Adds two float vectors on both the CPU and the GPU, then compares the results and the execution time of each. |
| `02_imageBlur` | Blurs an image by averaging each pixel's neighborhood on both the CPU and the GPU, and compares the execution time of each. |
| `03_simpleCNNInference` | Runs one-image-at-a-time CIFAR-10 CNN inference (Conv → ReLU → MaxPool → FC) with pre-trained weights, reporting Top-1 accuracy and per-layer timing. |
| `04_simpleCNNBatch` | Extends the same inference to batched execution, processing BATCH_SIZE images per kernel launch instead of one at a time. |
| `05_simpleCNNBatchLowering` | Replaces the direct convolution with explicit lowering (im2col + GEMM) and compares its timing against 04. |
| `06_simpleCNNBatchTiling` | Restructures the GEMM into a 2D tiled grid/block decomposition to improve locality, still reading from global memory. |
| `07_simpleCNNBatchTilingSharedMem` | Adds shared-memory caching to the tiled GEMM to cut down global memory traffic. |
| `08_largeConvolutionLayer` | Implements one large convolution layer step by step: direct convolution, im2col, and four GEMMs from flat to 2D tiled, shared memory, and Tensor Core. |
| `09_simpleCNNTraining` | Trains the CIFAR-10 CNN on the GPU with mini-batch SGD, implementing the backward and optimizer kernels. |
| `common` | Shared headers used by the exercises (timing, CIFAR-10 loader, CUDA utilities, network configuration). |
| `DB` | CIFAR-10 dataset, pre-trained weights, and the input image for the blur exercise. |

## Notes

- `01` through `04` ship with both the reference solution (`GPU`) and the student exercise (`GPUEmpty`).
- **From `05` on, no reference solution (`GPU`) is provided — students must fill in every blank in `GPUEmpty` themselves.**
