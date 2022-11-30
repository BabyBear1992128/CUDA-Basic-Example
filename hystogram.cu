/*
* CUDA Project(Histogram)
*
* Created on: 11.29.2022
* Author: BabyBear(babybearbear1280@gmail.com)
*
*/

// Include Heder File
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <device_functions.h>


// Define Consts
#define BLOCK_SIZE 512
#define NUM_BINS 4096
#define MAX_VAL 127
#define NUM_BINS 4096

/**
 * histogram_kernel computes the histogram of an input array on the GPU
 */
__global__ void histogram_kernel(unsigned int* input, unsigned int* bins, unsigned int numElements) {

    //@@ Insert code below to compute histogram of input using shared memory and atomics

    int tx = threadIdx.x; int bx = blockIdx.x;

    // compute global thread coordinates
    int i = (bx * blockDim.x) + tx;

    // create a private histogram copy for each thread block
    __shared__ unsigned int hist[NUM_BINS];

    // each thread must initialize more than 1 location
    if (NUM_BINS > BLOCK_SIZE) {
        for (int j = tx; j < NUM_BINS; j += BLOCK_SIZE) {
            if (j < NUM_BINS) {
                hist[j] = 0;
            }
        }
    }
    // use the first `NUM_BINS` threads of each block to init
    else {
        if (tx < NUM_BINS) {
            hist[tx] = 0;
        }
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // update private histogram
    if (i < numElements) {
        atomicAdd(&(hist[input[i]]), 1);
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // each thread must update more than 1 location
    if (NUM_BINS > BLOCK_SIZE) {
        for (int j = tx; j < NUM_BINS; j += BLOCK_SIZE) {
            if (j < NUM_BINS) {
                atomicAdd(&(bins[j]), hist[j]);
            }
        }
    }
    // use the first `NUM_BINS` threads to update final histogram
    else {
        if (tx < NUM_BINS) {
            atomicAdd(&(bins[tx]), hist[tx]);
        }
    }
}


/**
 * saturate at 127
 */
__global__ void convert_kernel(unsigned int* bins, unsigned int numBins) {
    //@@ Insert code below to clean up bins that saturate at 127
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < numBins) {
        if (bins[i] > MAX_VAL) {
            bins[i] = MAX_VAL;
        }
    }
}

/**
 * Main function
 */
int main(void) {
    // data params
    int inputLength;
    unsigned int* hostInput;
    unsigned int* hostBins;
    unsigned int* resultRef;
    unsigned int* deviceInput;
    unsigned int* deviceBins;



    //@@ Insert code below to read in inputLength from args
    printf("Please enter the length of the input array: ");
    scanf("%d", &inputLength);
    printf("\nThe input length is %d\n", inputLength);

    // determine size
    size_t histoSize = NUM_BINS * sizeof(unsigned int);
    size_t inSize = inputLength * sizeof(unsigned int);



    //@@ Insert code below to allocate Host memory for input and output
    hostInput = (unsigned int*)malloc(inSize);
    hostBins = (unsigned int*)malloc(histoSize);
    resultRef = (unsigned int*)malloc(inSize);



    //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    srand(clock());
    for (int i = 0; i < inputLength; i++) {
        hostInput[i] = int((float)rand() * (NUM_BINS - 1) / float(RAND_MAX));
    }


    //@@ Insert code below to create reference result in CPU
    for (int i = 0; i < inputLength; i++) {
        resultRef[i] = hostInput[i];
    }



    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void**)&deviceInput, inSize);
    cudaMalloc((void**)&deviceBins, histoSize);



    //@@ Insert code to Copy memory to the GPU here
    cudaMemset(deviceBins, 0, histoSize);



    // host2device transfer
    cudaMemcpy(deviceInput, hostInput, inSize, cudaMemcpyHostToDevice);



    //@@ Initialize the grid and block dimensions here
    dim3 threadPerBlock(BLOCK_SIZE, 1, 1);
    dim3 blockPerGrid(ceil(inputLength / (float)BLOCK_SIZE), 1, 1);


    //@@ Launch the GPU Kernel here
    histogram_kernel << <blockPerGrid, threadPerBlock >> > (deviceInput, deviceBins, inputLength);



    //@@ Initialize the second grid and block dimensions here
    threadPerBlock.x = BLOCK_SIZE;
    blockPerGrid.x = ceil(NUM_BINS / (float)BLOCK_SIZE);


    //@@ Launch the second GPU Kernel here
    convert_kernel << <blockPerGrid, threadPerBlock >> > (deviceBins, NUM_BINS);



    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, histoSize, cudaMemcpyDeviceToHost);



    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < inputLength; i++) {
        printf("output: %d, reference: %d \n", hostBins[i], resultRef[i]);
    }



    //@@ Free the GPU memory here
    free(hostBins);
    free(hostInput);



    //@@ Free the CPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);



    return 0;
}
