/*
 * =====================================================================================
 *
 *       Filename:  ppm.h
 *
 *    Description:  Minimal reader/writer for binary PPM (P6) image files,
 *                  used to load input images and save output images across
 *                  example codes.
 *
 *       Reference:  https://www.programmersought.com/article/75842998452/
 *                   https://www.cplusplus.com/reference/cstdio/fread/
 *
 *        Version:  1.0
 *        Created:  07/15/2021 12:53:21 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yoon, Myung Kuk, myungkuk.yoon@gmail.com
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#pragma once
#include<stdio.h>
#include<stdlib.h>

//MK: Loads a binary PPM (P6) file into a newly allocated RGB buffer
void ppmLoad(const char* filename, unsigned char** outData, int* w, int* h)
{
		char header[1024];
    int line = 0;
    FILE *fp = fopen(filename, "rb");

    while(line < 2){
        fgets(header, 1024, fp);
        if(header[0] != '#'){
            ++line;
        }
    }
 
    //MK: Width and height
    sscanf(header,"%d %d\n", w, h);

    //MK: Max pixel value line
    fgets(header, 20, fp);

    size_t pixSize = sizeof(unsigned char) * (*w) * (*h) * 3;
    printf("(Input File %s) Picture Size %d (height) x %d (width): Total Pixel %d\n", filename,  *h, *w, (*h)*(*w));
    *outData = (unsigned char *)malloc(pixSize);

    //MK: RGB pixel data
    size_t result = fread(*outData, (*w)*(*h)*3, 1, fp);

    fclose(fp);
}

//MK: Saves an RGB buffer as a binary PPM (P6) file
void ppmSave(const char* filename, unsigned char* inData, int w, int h)
{
    FILE *fp = fopen(filename, "wb");

    //MK: Header - format, width/height, max pixel value
    fprintf(fp,"P6\n%d %d\n255\n",w,h);

    //MK: RGB pixel data
    fwrite(inData, w*h*3, 1, fp);

    fclose(fp);
    
    printf("(Output File %s) Picture Size %d (height) x %d (width): Total Pixel %d\n", filename,  h, w, (h)*(w));
}


