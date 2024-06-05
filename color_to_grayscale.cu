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
    PPMImage* targetImage = (PPMImage*)malloc(sizeof(PPMImage));
    targetImage->data = (unsigned char*)malloc(imageSize);
    targetImage->width = image->width;
    targetImage->height = image->height;

    unsigned char* d_imageData = NULL;
    unsigned char* d_targetImageData = NULL;

    int imageSize = CHANNELS * sizeof(unsigned char) * (image->width * image->height);

    cudaMalloc(((void**)&d_imageData, size));
    cudaMalloc(((void**)&d_targetImageData, size));

    cudaMemcpy(d_imageData, image->data, size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int gridSizeX = (image->width + block_size-1) / block_size;
    int gridSizeY = (image->height + block_size-1) / block_size;
    dim3 gridSize(gridSizeX, gridSizeY, 1);
    dim3 blockSize(block_size, block_size, 1);
    
    colorToGrayScale<<<grid_size,block_size>>>(d_targetImageData, d_imageData, image->width, image->height);

    cudaMemcpy(targetImage->data, d_targetImageData, imageSize, cudaMemcpyDeviceToHost);

    writePPM("images/grayscale.ppm", targetImage);

    cudaFree(d_targetImageData);
    cudaFree(d_imageData);

    free(targetImage->data);
    free(targetImage);
    free(image->data);
    free(image);

    return 0;
}
