/*
* CUDA Project(Vector Addition)
* 
* Created on: 11.29.2022
* Author: BabyBear(babybearbear1280@gmail.com)
* 
*/


// Include Heder File
#include <stdio.h>
#include <time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// Define Data Type / to Double
#define DataType double

#define RAND_MAX 0x7fff
/**
* Addition Vectors
*
**/
__global__ void vecAdd(const DataType *A, const DataType *B, DataType *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}


//@@ Insert code to implement timer start


//@@ Insert code to implement timer stop




/**
 * Host main routine
 */
int main(void) {


  int numElements;
  DataType* hostInput1;
  DataType* hostInput2;
  DataType* hostOutput;
  DataType* resultRef;
  DataType* deviceInput1;
  DataType* deviceInput2;
  DataType* deviceOutput;


  //@@ Insert code below to read in inputLength from args
  printf("Please, input the count of elements: ");
  scanf("%d", &numElements);

  size_t size = numElements * sizeof(DataType);

  printf("The input length is %d\n", numElements);


  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *)malloc(size);
  hostInput2 = (DataType *)malloc(size);
  hostOutput = (DataType *)malloc(size);


  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < numElements; ++i) {
    hostInput1[i] = rand() / (DataType)RAND_MAX;
    hostInput2[i] = rand() / (DataType)RAND_MAX;
  }


  //@@ Insert code below to allocate GPU memory here
  deviceInput1 = NULL;
  cudaMalloc((void **)&deviceInput1, size);

  deviceInput2 = NULL;
  cudaMalloc((void **)&deviceInput2, size);

  deviceOutput = NULL;
  cudaMalloc((void **)&deviceOutput, size);


  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;



  //@@ Launch the GPU Kernel here
  // Start to begin timer
  clock_t start_d = clock();

  // Lanch the GPU Kernel
  vecAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, numElements);

  // Calculate time and output
  clock_t end_d = clock();
  clock_t start_h = clock();

  double time_d = (double)(end_d - start_d) / CLOCKS_PER_SEC;
      
  printf("Time: %fs\n\n", time_d);


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < numElements; ++i) {

    printf("[%d]: %f + %f = %f\n", i, hostInput1[i], hostInput2[i], hostOutput[i]);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  printf("Done\n");
  return 0;
}
