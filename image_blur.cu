#include <stdio.h>
#include <assert.h>

#include "utils.h"

#define CHANNELS 3
#define BLUR_SIZE 3

__global__ void imageBlur(unsigned char* pOut, unsigned char* pIn, int width, int height) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {

        int pixelValR = 0, pixelValG = 0, pixelValB = 0;
        int pixelCount = 0;

        // Get average of surrending pixels 
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; blurRow++) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; blurCol++) {
                int currRow = row + blurRow;
                int currCol = col + blurCol;
                int rgbOffset = (currRow*width+currCol)*CHANNELS;

                if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width) {
                    pixelValR += pIn[rgbOffset];
                    pixelValG += pIn[rgbOffset+1];
                    pixelValB += pIn[rgbOffset+2];

                    pixelCount++;
                }
            }
        }

        int offset = (row * width + col) * CHANNELS;
        pOut[offset] = (unsigned char)(pixelValR / (float)pixelCount);
        pOut[offset+1] = (unsigned char)(pixelValG / (float)pixelCount);
        pOut[offset+2] = (unsigned char)(pixelValB / (float)pixelCount);
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
    
    imageBlur<<<gridSize,blockSize>>>(d_targetImageData, d_imageData, image->width, image->height);

    cudaMemcpy(targetImage->data, d_targetImageData, imageSize, cudaMemcpyDeviceToHost);
    printf("targetImage->data %d\n", targetImage->data[0]);
    

    printf("writing images");
    writePPM("images/blur.ppm", targetImage);

    cudaFree(d_targetImageData);
    cudaFree(d_imageData);

    free(targetImage->data);
    free(targetImage);
    free(image->data);
    free(image);

    return 0;
}
