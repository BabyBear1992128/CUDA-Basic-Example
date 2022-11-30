/*
* CUDA Project(Multiple Matrix)
*
* Created on: 11.29.2022
* Author: BabyBear(babybearbear1280@gmail.com)
*
*/

// Include Heder File
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define Data Type / to Double
#define DataType double

#define BLOCK_SIZE 16

/**
* Multiplication Matrix
    m X n matrix (A)
    n X k matrix (B)
    m X k matrix (C)
**/
__global__ void gpu_matrix_mult(DataType* a, DataType* b, DataType* c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

/**
 * Main function
 */
int main(void) {

    // Define variables
    DataType* hostA; // The A matrix
    DataType* hostB; // The B matrix
    DataType* hostC; // The output C matrix
    DataType* hostCC; // The output C matrix
    DataType* deviceA;
    DataType* deviceB;
    DataType* deviceC;

    /*
     a m X n matrix (A)
     a n X k matrix (B)
     a m X k matrix (C)
    */
    int m, n, k;


    //@@ Insert code below to read in m, n, k
    srand(3333);
    printf("please, input : \"m n k\"\n");
    scanf("%d %d %d", &m, &n, &k);


    //@@ Insert code below to allocate Host memory for input and output
    cudaMallocHost((void**)&hostA, sizeof(int) * m * n);
    cudaMallocHost((void**)&hostB, sizeof(int) * n * k);
    cudaMallocHost((void**)&hostC, sizeof(int) * m * k);
    cudaMallocHost((void**)&hostCC, sizeof(int) * m * k);


    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    for (int i = 0; i < m; ++i) { // matrix B
        for (int j = 0; j < n; ++j) {
            hostA[i * n + j] = rand() + 1024 / 13;
            printf("%f", hostA[i * n + j]);
        }
    }

    for (int i = 0; i < n; ++i) { // matrix B
        for (int j = 0; j < k; ++j) {
            hostB[i * k + j] = rand() + 1024 / 13;
        }
    }




    // @@ Insert code below to allocate GPU memory here
    cudaMalloc((void**)&deviceA, sizeof(int) * m * n);
    cudaMalloc((void**)&deviceB, sizeof(int) * n * k);
    cudaMalloc((void**)&deviceC, sizeof(int) * m * k);


    //@@ Insert code to below to Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(int) * n * k, cudaMemcpyHostToDevice);


    //@@ Initialize the grid and block dimensions here
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grideviceCols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grideviceCols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


    //@@ Launch the GPU Kernel here
    gpu_matrix_mult << <dimGrid, dimBlock >> > (deviceA, deviceB, deviceC, m, n, k);


    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();


    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            printf("C[%d][%d]:%f, ", i, j, hostC[i * k + j]);
        }
        printf("\n");
    }


    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);


    //@@ Free the CPU memory here
    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);
    cudaFreeHost(hostCC);


    return 0;
}
