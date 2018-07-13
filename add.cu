#include <iostream>
#include <math.h>

using namespace std;


__global__
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main() {

    //For my own sanity lets explain this.
    //1<<20 is a notation that in this context represents
    //a bitshift. That means that you have the bit 1 and then you shift it to the
    //(in this case) left by 20 spaces and fill the empty space with zeros.
    int N = 1<<30; // 1M elements

    float *x, *y;

    //Allocates "Unified Memory" which is accessible from both the CPU and GPU.
    cudaError_t err = cudaMallocManaged(&x, N*sizeof(float));
    if (err != cudaSuccess) {
	cout << "CUDA Error" << endl;
   	//printf("%s\n", cudaGetErrorString(err));
    }	 
   // cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    //initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    //Runs cuda kernel on 1M elements on the CPU
    add<<<1, 1>>>(N, x, y);

    cout << "Add Completed" << endl;

    //Forces CPU to wait for GPU to finish before accessing
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
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
