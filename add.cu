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
    	int remainder = arraySize % stride;
	if (remainder == 0) {
	    valuesPerThread = arraySize / stride;
	} else {
	    // This is not perfectly optimized, but it will do for now.
	    valuesPerThread = floor(arraySize / stride) + remainder;
	}

    } else {
    	valuesPerThread = 1;
    }

    int correctedIndex = index-1;

    if (index != 1) {
    	correctedIndex = index + valuesPerThread;
    } else {
	correctedIndex = index;
    } 

    // Each threads will iterate through all assigned values
    for (int i = 0; i < valuesPerThread; i++) {
        dY[correctedIndex+i] = dX[correctedIndex+i] + dY[correctedIndex+i];
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
