#include <stdio.h>
#include <assert.h>

#include "utils.h"

#define CHANNELS 3

__global__ void colorToGrayScale(unsigned char* pOut, unsigned char* pIn, int width, int height) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // offset into the data
        int grayOffset = row * width + col;

        // RGB has CHANNELS more columns than gray scale. 
        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = pIn[rgbOffset];
        unsigned char g = pIn[rgbOffset+1];
        unsigned char b = pIn[rgbOffset+2];

        unsigned char outputPixel = 0.21f*r + 0.71*g + 0.07*b;

        // Using PPM format so just output rgb the same values. 
        pOut[rgbOffset] = outputPixel;
        pOut[rgbOffset+1] = outputPixel;
        pOut[rgbOffset+2] = outputPixel;
    }
}

int main() {
    PPMImage* image = readPPM("images/image.ppm");
    if (image->data == NULL)
    {
        printf("Image is NULL!");
        return 0;
    }

    PPMImage* targetImage = (PPMImage*)malloc(sizeof(PPMImage));
    int imageSize = CHANNELS * sizeof(unsigned char) * (image->width * image->height);

    targetImage->data = (unsigned char*)malloc(imageSize);
    targetImage->width = image->width;
    targetImage->height = image->height;

    unsigned char* d_imageData = NULL;
    unsigned char* d_targetImageData = NULL;

    cudaMalloc((void**)&d_imageData, imageSize);
    cudaMalloc((void**)&d_targetImageData, imageSize);

    cudaMemcpy(d_imageData, image->data, imageSize, cudaMemcpyHostToDevice);

    int block_size = 16;
    int gridSizeX = (image->width + block_size-1) / block_size;
    int gridSizeY = (image->height + block_size-1) / block_size;
    printf("Grid size: %d, %d\n", gridSizeX, gridSizeY);

    dim3 gridSize(gridSizeX, gridSizeY, 1);
    dim3 blockSize(block_size, block_size, 1);
    printf("Image size: %d, %d\n", image->width, image->height);
    
    colorToGrayScale<<<gridSize,blockSize>>>(d_targetImageData, d_imageData, image->width, image->height);

    cudaMemcpy(targetImage->data, d_targetImageData, imageSize, cudaMemcpyDeviceToHost);
    printf("targetImage->data %d\n", targetImage->data[0]);
    

    printf("writing images");
    writePPM("images/grayscale.ppm", targetImage);

    cudaFree(d_targetImageData);
    cudaFree(d_imageData);

    free(targetImage->data);
    free(targetImage);
    free(image->data);
    free(image);

    return 0;
}
