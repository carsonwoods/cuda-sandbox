#include <iostream>
#include <stdio.h>

using namespace std;

__global__ void add(float *dX, float *dY) {

    // contains the index of the current thread in the block
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    

    valuesPerThread = 1;


    // Each threads will iterate through all assigned values
    for (int i = 0; i < valuesPerThread; i++) {
        dY[index+i] = dX[index+i] + dY[index+i];
    }
    
    
}


int main() {
    
    int N = 256;
    int memSize = N*sizeof(float);
    
    float x[N], y[N], z[N];
    
    float *dX, *dY;
    
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 50.0f;
    }
    
    cudaMalloc(&dX, memSize);
    cudaMalloc(&dY, memSize);
    
    cudaMemcpy(dX, &x, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, &y, memSize, cudaMemcpyHostToDevice);

    
    
    add<<<1, 50>>>(dX, dY);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(&z, dY, memSize, cudaMemcpyDeviceToHost);
    
    cudaFree(dX);
    cudaFree(dY);

    printf("Done!\n");
    printf("Memory freed\n");
    
    return 0;
    
}
