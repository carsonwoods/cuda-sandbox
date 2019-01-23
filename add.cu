#include <iostream>
#include <stdio.h>

using namespace std;

__global__ void add(float *dX, float *dY, int N) {

    // contains the index of the current thread in the block
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int arraySize = N;
    int valuesPerThread;


    if (stride < arraySize) {
        // Gets the amount of values not assigned to a thread
    	int remainder = arraySize % stride;
    	
        // This is not perfectly optimized, but it will do for now.
        valuesPerThread = floor(arraySize / stride) + remainder;

    } else {
    	valuesPerThread = 1;
    }


    // Assigns a range of values to each thread
    startLocation = index*valuesPerThread;


    // Each threads will iterate through all assigned values
    for (int i = 0; i < valuesPerThread; i++) {
        dY[startLocation+i] = dX[startLocation+i] + dY[startLocation+i];
    } 
}


int main() {
    
    int N = 256;
    int memSize = N*sizeof(float);
    
    float x[N], y[N], z[N];
    
    float *dX = (float *) malloc(memSize);
    float *dY = (float *) malloc(memSize);

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 40.0f;
    }
    
    cudaMalloc(&dX, memSize);
    cudaMalloc(&dY, memSize);
    
    cudaMemcpy(dX, &x, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, &y, memSize, cudaMemcpyHostToDevice);

    
    
    add<<<1, 50>>>(dX, dY, N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(&z, dY, memSize, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
    	cout << z[i] << " - " << i << endl;
    }

    cudaFree(dX);
    cudaFree(dY);

    printf("Done!\n");
    printf("Memory freed\n");
    
    return 0;
    
}
