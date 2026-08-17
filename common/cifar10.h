/*
 * =====================================================================================
 *
 *       Filename:  cifar10.h
 *
 *    Description:  Loader for the CIFAR-10 binary batch format (as
 *                  distributed in cifar-10-batches-bin): each record is
 *                  1 label byte followed by 3072 image bytes laid out as
 *                  3 consecutive 32x32 channel planes (R, then G, then
 *                  B) - i.e. already in (C, H, W) order.
 *
 *       Reference:  https://www.cs.toronto.edu/~kriz/cifar.html
 *
 *        Version:  1.0
 *        Created:  07/14/2026 03:10:00 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#pragma once
#include <stdio.h>
#include <stdlib.h>

#define CIFAR_IMAGE_C 3
#define CIFAR_IMAGE_H 32
#define CIFAR_IMAGE_W 32
#define CIFAR_IMAGE_SIZE (CIFAR_IMAGE_C * CIFAR_IMAGE_H * CIFAR_IMAGE_W)
#define CIFAR_RECORD_SIZE (CIFAR_IMAGE_SIZE + 1)

//MK: Loads a CIFAR-10 *.bin batch file into freshly allocated label/image arrays
void cifar10Load(const char* filename, unsigned char** labels, unsigned char** images, int* numImages){
	FILE *fp = fopen(filename, "rb");

	fseek(fp, 0, SEEK_END);
	long fileSize = ftell(fp);
	rewind(fp);

	int n = (int)(fileSize / CIFAR_RECORD_SIZE);
	*numImages = n;
	*labels = (unsigned char*)malloc((size_t)n);
	*images = (unsigned char*)malloc((size_t)n * CIFAR_IMAGE_SIZE);

	for(int i = 0; i < n; i++){
		fread(*labels + i, 1, 1, fp);
		fread(*images + (size_t)i * CIFAR_IMAGE_SIZE, 1, CIFAR_IMAGE_SIZE, fp);
	}

	fclose(fp);
	printf("(CIFAR-10 %s) Loaded %d images\n", filename, n);
}

//MK: Loads and concatenates multiple CIFAR-10 *.bin batch files (e.g. all 5 training batches) into one label/image array
void cifar10LoadMultiple(const char **filenames, int numFiles, unsigned char **labels, unsigned char **images, int *numImages){
	int total = 0;
	for(int f = 0; f < numFiles; f++){
		FILE *fp = fopen(filenames[f], "rb");
		fseek(fp, 0, SEEK_END);
		long fileSize = ftell(fp);
		fclose(fp);
		total += (int)(fileSize / CIFAR_RECORD_SIZE);
	}

	*numImages = total;
	*labels = (unsigned char*)malloc((size_t)total);
	*images = (unsigned char*)malloc((size_t)total * CIFAR_IMAGE_SIZE);

	int offset = 0;
	for(int f = 0; f < numFiles; f++){
		FILE *fp = fopen(filenames[f], "rb");
		fseek(fp, 0, SEEK_END);
		long fileSize = ftell(fp);
		rewind(fp);
		int n = (int)(fileSize / CIFAR_RECORD_SIZE);

		for(int i = 0; i < n; i++){
			fread(*labels + offset + i, 1, 1, fp);
			fread(*images + (size_t)(offset + i) * CIFAR_IMAGE_SIZE, 1, CIFAR_IMAGE_SIZE, fp);
		}
		fclose(fp);
		printf("(CIFAR-10 %s) Loaded %d images\n", filenames[f], n);
		offset += n;
	}
}
