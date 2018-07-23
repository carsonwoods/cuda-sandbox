#include <iostream>
#include <math.h>

using namespace std;


__global__
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
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
     for (int i = 0; i < n; i++) {
         z[i] = (a * x[i]) + (b * y[i]);
     }
}


//This is called a vertical operation
//Will find the largest element in a vector (in this case vector of floats)
//Single threaded on GPU.
__global__
float verticalOperation(int n, float* x) {
    float largestValue = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > largestValue) {
            largestValue = x[i];
        }
    }
    return largestValue;
}


int main() {

    //For my own sanity lets explain this.
    //1<<20 is a notation that in this context represents
    //a bitshift. That means that you have the bit 1 and then you shift it to the
    //(in this case) left by 20 spaces and fill the empty space with zeros.
    int N = 1<<20; // 1M elements

    float *x, *y;

    //Allocates "Unified Memory" which is accessible from both the CPU and GPU.
    cudaError_t cudaMallocErr1 = cudaMallocManaged(&x, N*sizeof(float));
    if (cudaMallocErr1 != cudaSuccess) {
	    cout << "CUDA Error" << endl;
    }
    cudaError_t cudaMallocErr2 = cudaMallocManaged(&y, N*sizeof(float));
    if (cudaMallocErr2 != cudaSuccess) {
        cout << "CUDA Error" << endl;
    }  	 

    //initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    //Runs cuda kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);
    add2<<<1, 1>>>(N, x, y, 4.0, 5.0);
    cout <<  verticalOperation<<<1, 1>>>(N, x) << endl;

    cout << "Done!" << endl;

    //Forces CPU to wait for GPU to finish before accessing
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    //NOTE: This does not work for any function other than add()
    //NOTE: This is a massive performance hit. Will eventually delegate to GPU.
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
} 
