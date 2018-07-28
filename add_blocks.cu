#include <iostream>
#include <math.h>
#include <stdio.h>

using namespace std;


__global__
void add(int n, float *x, float *y) {
    //contains the index of the current thread within its block
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    //contins the number of threads in the block
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}


//I don't relly know what to call this function so this describes its
//functionality and it will be named add2 because its basically add + extra stuff
//z = alpha*x + beta* y
//where z, x, and y are vectors of length N, and alpha and beta are scalars.
__global__
void add2(int n, float *x, float *y, float a, float b) {
    float *z = new float[n];

    //contains the index of the current thread within its block
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    //contains the number of threads in the block
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        z[i] = (a * x[i]) + (b * y[i]);
    }
}

__global__ void verticalOperation(int size, float *global_input_data, float *global_output_data) {
    printf("%d\n",global_input_data[5]);

    //each thread loads one element from global memory into shared memory
    int thread_id = threadIdx.x;
    int numBlocks = gridDim.x;
    int numTotalThreads = gridDim.x * blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float thread_maxima[];

    //Sets each thread's starting point in the global_input_data
    int loopIndex = (size/numTotalThreads) + index;

    //sets thread_max to first element in array for initial comparison
    float thread_max = global_input_data[loopIndex];

    //iterate across a threads domain until maximum is found
    //Loop operates as follows:
    //Each thread has its starting position set by loopIndex.
    //Loop will iterate until the starting point of the next thread is hit.
    //Each iteration the thread compares the value in the array against the current thread_max
    for(int i = loopIndex; i < loopIndex + size/blockDim.x; i++) {
        if (global_input_data[loopIndex + i] > thread_max) {
            thread_max = global_input_data[loopIndex + i];
        }
    }

    //max for each thread is placed into thread_maxmia for next comparison
    //NOTE: Since thread_maxima is shared memory thread_id is used.
    //      Shared Memory is only shared across a single block so index isn't used.
    thread_maxima[thread_id] = thread_max;

    //Threads are synced to ensure all comparisons that need to be made are done.
    __syncthreads();

    //find the maximum value from all threads in a single block
    //appends that value to block_maximum[]
    if (thread_id == 0) {
        float max = thread_maxima[0];

        //thread 0 of each block iterates across all values of thread_maxima
        //numTotalThreads/(size/blockDim.x) should be the size of thread_maxima
        for (int x = 0; x < numTotalThreads/(size/blockDim.x); x++) {
            if (thread_maxima[x] > max) {
                max = thread_maxima[x];
            }
        }

        //The largest value of each block is placed in global_output_data
        global_output_data[blockIdx.x] = max;
    }


    __syncthreads();

    //find the maximum value from all blocks using one last thread
    if (loopIndex == 0) {
        float max = global_output_data[0];
        for (int x = 0; x < numBlocks; x++) {
            if (global_output_data[x] > max) {
                max = global_output_data[x];
            }
        }
        global_output_data[0] = max;
    }

}

void testVerticalOperation() {
    //For my own sanity lets explain this.
    //1<<20 is a notation that in this context represents
    //a bitshift. That means that you have the bit 1 and then you shift it to the
    //(in this case) left by 20 spaces and fill the empty space with zeros.
    int N = 1<<18; // 1M elements

    float *deviceArray;
    float hostArray[N], result[N];

    //Allocates "Unified Memory" which is accessible from both the CPU and GPU.
    cudaError_t cudaMallocErr1 = cudaMallocManaged(&deviceArray, N*sizeof(float));
    if (cudaMallocErr1 != cudaSuccess) {
        cout << "CUDA Error" << endl;
    }


    //initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        hostArray[i] = 1.0f;
        result[i] = 0.0f;
    }

    int blockSize = 256;
    int numBlocks = N/blockSize;


    //ensures that there is a value that could be largest
    t[5] = 987654.0f;

    //copy memory to device from host and print error if found
    cudaError_t cudaMemcpy1Err = cudaMemcpy(deviceArray, hostArray, N*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaMemcpy1Err != cudaSuccess) {
        cout << "Memcpy to Device Error: " << cudaMemcpy1Err << endl;
    }

    verticalOperation<<<numBlocks, blockSize, N*sizeof(float)>>>(N, x, z);

    //copy memory to host from device and print error if found
    cudaError_t cudaMemcpy2Err = cudaMemcpy(result, deviceArray, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaMemcpy2Err != cudaSuccess) {
        cout << "Memcpy to Host Error: " << cudaMemcpy2Err << endl;
    }

    cout << "Largest value in hostArray: " << result[0] << endl;

    cout << "Done!" << endl;

    //Forces CPU to wait for GPU to finish before accessing
    cudaDeviceSynchronize();

    // Free memory
    cudaFree(deviceArray);
    
    return 0;

}


int main() {

    //For my own sanity lets explain this.
    //1<<20 is a notation that in this context represents
    //a bitshift. That means that you have the bit 1 and then you shift it to the
    //(in this case) left by 20 spaces and fill the empty space with zeros.
    int N = 1<<18; // 1M elements

    float *x, *y, *z;
    float t[N], q[N];

    //Allocates "Unified Memory" which is accessible from both the CPU and GPU.
    cudaError_t cudaMallocErr1 = cudaMallocManaged(&x, N*sizeof(float));
    if (cudaMallocErr1 != cudaSuccess) {
        cout << "CUDA Error" << endl;
    }
    cudaError_t cudaMallocErr2 = cudaMallocManaged(&y, N*sizeof(float));
    if (cudaMallocErr2 != cudaSuccess) {
        cout << "CUDA Error" << endl;
    }
    cudaError_t cudaMallocErr3 = cudaMallocManaged(&z, N*sizeof(float));
    if (cudaMallocErr3 != cudaSuccess) {
        cout << "CUDA Error" << endl;
    }

    //initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        y[i] = 2.0f;
        t[i] = 1.0f;
        q[i] = 0.0f;
    }

    int blockSize = 256;
    int numBlocks = N/blockSize;

    //add<<<numBlocks, blockSize>>>(N, x, y);
    //add2<<<numBlocks, blockSize>>>(N, x, y, 4.0, 5.0);

    //ensures that there is a value that could be largest
    t[5] = 987654.0f;

    //copy memory to device from host and print error if found
    cudaError_t cudaMemcpy1Err = cudaMemcpy(x, t, N*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaMemcpy1Err != cudaSuccess) {
        cout << "Memcpy to Device Error: " << cudaMemcpy1Err << endl;
    }

    verticalOperation<<<numBlocks, blockSize, N*sizeof(float)>>>(N, x, z);

    //copy memory to host from device and print error if found
    cudaError_t cudaMemcpy2Err = cudaMemcpy(q, x, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaMemcpy2Err != cudaSuccess) {
        cout << "Memcpy to Host Error: " << cudaMemcpy2Err << endl;
    }

    cout << "Largest value in array q: " << q[0] << endl;

    cout << "Done!" << endl;

    //Forces CPU to wait for GPU to finish before accessing
    cudaDeviceSynchronize();

    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;

}
