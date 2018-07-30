#include <iostream>
#include <stdio.h>
#include <cstdlib>

using namespace std;

__global__ void verticalOperation(int size, float *deviceArray, float *deviceResult) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x; //NOTE: this would also be the amount of values in shared memory (or it should be anyways)

    //allocated shared memory to reduce global memory access overhead
    extern __shared__ float sdata[];

    //move each value from deviceArray pointer into shared_memory_array
    sdata[threadIdx.x] = deviceArray[index];

    __syncthreads();
    
    //stride is currently the length of the unsorted array that still needs to be compared
    for (int stride = blockSize; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride/2) {
            if (sdata[threadIdx.x + (stride/2)] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + (stride/2)];
            }
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) { deviceResult[blockIdx.x] = sdata[0]; }

    //stride is currently the length of the unsorted array that still needs to be compared
    for (int stride = gridDim.x; stride >= 1; stride /= 2) {
        if (index < stride/2) {
            if (deviceResult[index + (stride/2)] > deviceResult[index]) {
                deviceResult[index] = deviceResult[index + (stride/2)];
            }
        }
    }
}

void testVerticalOperation() {

    int number_of_values = 1 << 18;

    int memSize = number_of_values*sizeof(float);

    int blockSize = 256;
    int numBlocks = 1024;

    float *deviceValue, *deviceResult; //device copies
    float initialValue[number_of_values], result[number_of_values]; //host copies

    for (int x = 0; x < number_of_values; x++) {
        initialValue[x] = 0.0f;
    }

    initialValue[2] = 500.0f;
    initialValue[3] = 600.0f;
    initialValue[66] = 998.0f;
    initialValue[30000] = 1000.0f;


    //Allocates "Unified Memory" which is accessible from both the CPU and GPU.
    cudaError_t cudaMallocErr1 = cudaMalloc(&deviceValue, memSize);
    if (cudaMallocErr1 != cudaSuccess) {
        cout << "CUDA Error" << endl;
    }

    //Allocates "Unified Memory" which is accessible from both the CPU and GPU.
    cudaError_t cudaMallocErr2 = cudaMalloc(&deviceResult, memSize);
    if (cudaMallocErr2 != cudaSuccess) {
        cout << "CUDA Error" << endl;
    }


    //copy memory to device from host and print error if found
    cudaError_t cudaMemcpy1Err = cudaMemcpy(deviceValue, &initialValue, memSize, cudaMemcpyHostToDevice);
    if (cudaMemcpy1Err != cudaSuccess) {
        cout << "Memcpy to Device Error: " << cudaMemcpy1Err << endl;
    }

    verticalOperation<<<numBlocks, blockSize, memSize/blockSize>>>(number_of_values, deviceValue, deviceResult);

    //Forces CPU to wait for GPU to finish before accessing
    cudaDeviceSynchronize();


    //copy memory to host from device and print error if found
    cudaError_t cudaMemcpy2Err = cudaMemcpy(&result, deviceResult, memSize, cudaMemcpyDeviceToHost);
    if (cudaMemcpy2Err != cudaSuccess) {
        cout << "Memcpy to Host Error: " << cudaMemcpy2Err << endl;
    }

    cout << result[0] << endl;

    cout << "Done!" << endl;

    // Free memory
    cudaFree(deviceValue);
    cudaFree(deviceResult);
}


int main() {
    //Runs test for verticalOperation kernal on GPU
    testVerticalOperation();

    return 0;

}
