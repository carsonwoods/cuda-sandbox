#include <iostream>
#include <stdio.h>

using namespace std;

__global__ void verticalOperation(int size, float *deviceArray, float *deviceResult) {
    
    int numBlocks = gridDim.x;
    int numTotalThreads = gridDim.x * blockDim.x;
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
      
    
    float max = deviceArray[0];
    for (int x = 0; x < size; x++) {
        if (deviceArray[x] > max) {
            max = deviceArray[x];
        }
    }
    
    deviceResult[0] = max;
    
    /*
    extern __shared__ float thread_maxima[];
    
    //Sets each thread's starting point in the deviceArray
    int loopIndex = (size/numTotalThreads) * thread_index;

    //sets thread_max to first element in array for initial comparison
    float thread_max = deviceArray[loopIndex];
    
    //iterate across a threads domain until maximum is found
    //Loop operates as follows:
    //Each thread has its starting position set by loopIndex.
    //Loop will iterate until the starting point of the next thread is hit.
    //Each iteration the thread compares the value in the array against the current thread_max
    for(int i = loopIndex; i < loopIndex + size/blockDim.x; i++) {
        if (deviceArray[loopIndex + i] > thread_max) {
            thread_max = deviceArray[loopIndex + i];
        }
    }
    

    
    //max for each thread is placed into thread_maxmia for next comparison
    //NOTE: Since thread_maxima is shared memory thread_id is used.
    //      Shared Memory is only shared across a single block so index isn't used.
    thread_maxima[thread_index] = thread_max;
    
    //Threads are synced to ensure all comparisons that need to be made are done.
    __syncthreads();
    
    
    //find the maximum value from all threads in a single block
    //appends that value to block_maximum[]
    if (threadIdx.x == 0) {
        float max = thread_maxima[0];
        //thread 0 of each block iterates across all values of thread_maxima
        //numTotalThreads/(size/blockDim.x) should be the size of thread_maxima
        for (int x = 0; x < blockDim.x; x++) {
            if (thread_maxima[x] > max) {
                max = thread_maxima[x];
            }
        }

        //The largest value of each block is placed in deviceArray
        deviceResult[blockIdx.x] = max;
    }
    
    //find the maximum value from all blocks using one last thread
        
    if (loopIndex == 0) {
        float max = deviceResult[0];
        for (int x = 1; x < numBlocks; x++) {
            if (deviceResult[x] > max) {
                max = deviceResult[x];
            }
        }
        deviceResult[0] = max;
    }
    
    */
    
}

void testVerticalOperation() {

    int number_of_values = 1 << 18;
    
    int memSize = number_of_values*sizeof(float);
    
    int blockSize = 256;
    int numBlocks = 1024;
    
    float *deviceValue, *deviceResult; //device copies
    float initialValue[number_of_values], result[number_of_values]; //host copies

    for (int x = 0; x < number_of_values; x++) {
        initialValue[x] = 289.0f;
    }
    
    initialValue[2] = 500.0f;
     
     
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

    verticalOperation<<<1, 1>>>(number_of_values, deviceValue, deviceResult);

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
  




    //cudaDeviceSynchronize(); 
    return 0;

}
