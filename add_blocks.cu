#include <iostream>
#include <stdio.h>

using namespace std;

__global__ void testPrint() {
    printf("Hello world\n");
}

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

int main() {

    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0;
}
