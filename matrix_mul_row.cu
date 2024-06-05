#include <stdio.h>
#include <assert.h>

#define CHANNELS 3
#define BLUR_SIZE 3

#define N 100

__global__ void matrixMul(float* iM, float* iN, float* oP, int width) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width)
    {
        for (int col = 0; col < width; col++)
        {
            float sum = 0;
            for (int k = 0; k < width; k++)
            {
                sum += iM[row*width+k]*iN[k*width+col];
            }

            oP[row*width+col] = sum;
        }
    }
}

int main() {

    float* h_M = NULL;
    float* h_N = NULL;
    float* h_P = NULL;

    float* d_M = NULL;
    float* d_N = NULL;
    float* d_P = NULL;

    h_M = (float*)malloc(N*N*sizeof(float));
    h_N = (float*)malloc(N*N*sizeof(float));
    h_P = (float*)malloc(N*N*sizeof(float));

    for (int i = 0; i < N*N; i++)
    {
        h_M[i] = i % N;
        h_N[i] = (i+1) % N;
    }

    cudaMalloc((void**)&d_M, N*N*sizeof(float));
    cudaMalloc((void**)&d_N, N*N*sizeof(float));
    cudaMalloc((void**)&d_P, N*N*sizeof(float));

    cudaMemcpy(d_M, h_M, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, N*N*sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 2;
    int gridSizeX = (N + block_size-1) / block_size;
    //int gridSizeY = (N + block_size-1) / block_size;
    //printf("Grid size: %d, %d\n", gridSizeX, gridSizeY);

    //dim3 gridSize(gridSizeX, gridSizeY, 1);
    //dim3 blockSize(block_size, block_size, 1);
    
    matrixMul<<<gridSizeX,block_size>>>(d_M, d_N, d_P, N);

    cudaMemcpy(h_P, d_P, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // verify
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += h_M[i*N+k] * h_N[k*N+j];
            }
            assert(fabs(h_P[i*N+j]-sum) <= 1e-6);
        }
    }

    cudaFree(d_P);
    cudaFree(d_N);
    cudaFree(d_M);

    free(h_P);
    free(h_N);
    free(h_M);

    return 0;
}
