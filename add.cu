#include <iostream>
#include <stdio.h>

using namespace std;

__global__ void add(float *dX, float *dY, int N) {

    // contains the index of the current thread in the block
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int arraySize = N;
    int valuesPerThread;

    int remainder = 0;
    
    if (stride < arraySize) {
        // Gets the amount of values not assigned to a thread
        remainder = arraySize % stride;
    	
        // Determines how many values each thread should add.
        valuesPerThread = int(arraySize / stride);

        // Checks to see if casting rounds up and corrects
        if (valuesPerThread > (arraySize/stride)) {
            valuesPerThread = valuesPerThread - 1;
        }
    } else {
    	valuesPerThread = 1;
    }

    // Assigns a range of values to each thread
    int startLocation = index*valuesPerThread;


    // Each threads will iterate through all assigned values
    for (int i = 0; i < valuesPerThread; i++) {
        dY[startLocation+i] = dX[startLocation+i] + dY[startLocation+i];
    } 

    // Takes a remaining thread and sequentially handles any leftover values.
    if (index == 0 && remainder != 0) {
        // Remainder index stores the starting position of un-added values
        int remainderIndex = arraySize - remainder - 1;
        for (int i = 1; i <= remainder; i++) {
            dY[remainderIndex + i] = dX[remainderIndex + i] + dY[remainderIndex + i];
        }
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
    
    int errorCount = 0;
    for (int i = 0; i < N; i++) {
        if (z[i] != 41) {
            errorCount += 1;
        }
    }

    int percentError = (errorCount/N)*100

    printf("Percent Error: %d", percentError);

    cudaFree(dX);
    cudaFree(dY);

    printf("Done!\n");
    printf("Memory freed\n");
    
    return 0;
    
}
