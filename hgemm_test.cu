#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#define INDEX(i,j,ld) (((j)*(ld))+(i))

// Uncomment below to print debug statements
// #define DEBUG

using namespace std;


int hgemm_test(int M, int N) {

    
    int i, j;              // Array index iterators
    
    cudaError_t error;
    
    cublasStatus_t status;
    cublasHandle_t handle;
    
    float* arrayA = 0;     // Will store a copy of the array on the host device
    float* arrayB = 0;     // Will store a copy of the array on the host device
    float* arrayC = 0;     // Will store a copy of the array on the host device

    float* deviceArrayA;   // Will store a copy of the array on the device
    float* deviceArrayB;   // Will store a copy of the array on the device
    float* deviceArrayC;   // Will store a copy of the array on the device

    arrayA = (float *)malloc(M * N * sizeof(arrayA));
    arrayB = (float *)malloc(M * N * sizeof(arrayB));
    arrayC = (float *)malloc(M * N * sizeof(arrayC));
    
    // Ensures that host memory is allocated.
    if (!arrayA || !arrayB || !arrayC) {
        printf("Host memory allocation failed\n");
        return -1;
    }
    
    // Fills all Host Arrays with values
    for (j = 0; j < N; j++) {
        // Iterates Through Rows
        for (i = 0; i < M; i++) {
            // Iterates Through Columns
            arrayA[INDEX(i,j,M)] = (float)(i * M + j + 1);
            arrayB[INDEX(i,j,M)] = (float)(i * M + j + 2);
            arrayC[INDEX(i,j,M)] = 0;
        }
    }
    
    error = cudaMalloc((void**)&deviceArrayA, M*N*sizeof(*arrayA));
    if (error != cudaSuccess) {
        printf("Device Memory Allocation Failed\n");
        free(arrayA);
        free(arrayB);
        free(arrayC);
        return -1;
    }    

    
    error = cudaMalloc((void**)&deviceArrayB, M*N*sizeof(*arrayB));
    if (error != cudaSuccess) {
        printf("Device Memory Allocation Failed\n");
        cudaFree(deviceArrayA);
        free(arrayA);
        free(arrayB);
        free(arrayC);
        return -1;
    }
    
    error = cudaMalloc((void**)&deviceArrayC, M*N*sizeof(*arrayC));
    if (error != cudaSuccess) {
        printf("Device Memory Allocation Failed\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        free(arrayA);
        free(arrayB);
        free(arrayC);
        return -1;
    }
    
    #ifdef DEBUG
        printf("Device Memory Allocated Successfully\n");
    #endif    

    // Initialize CUBLAS API
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API Initialization Failed\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        cudaFree(deviceArrayC);
        free(arrayA);
        free(arrayB);
        free(arrayC);
        return -1;
    }
    
    #ifdef DEBUG
        printf("cuBLAS API Initialized Successfully\n");
    #endif
    
    // Move Matrix A from Host to Device
    // NOTE: M = Row count & Leading Dimension A/B(lda/ldb)
    // NOTE: N = Col count
    status = cublasSetMatrix(M, N, sizeof(*arrayA), arrayA, M, deviceArrayA, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: Failed to move data to device.\n");    
        printf("Freeing Memory and exiting\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        cudaFree(deviceArrayC);
        free(arrayA);
        free(arrayB);
        free(arrayC);
        cublasDestroy(handle);
        return -1;
    }
    
    // Move Matrix B from Host to Device
    // NOTE: M = Row count & Leading Dimension A/B(lda/ldb)
    // NOTE: N = Col count
    status = cublasSetMatrix(M, N, sizeof(*arrayB), arrayB, M, deviceArrayB, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: Failed to move data to device.\n");
        printf("Freeing Memory and exiting\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        cudaFree(deviceArrayC);

        free(arrayA);
        free(arrayB);

        free(arrayC);
        cublasDestroy(handle);
        return -1;
    }
    
    
    // Move Matrix B from Host to Device
    // NOTE: M = Row count & Leading Dimension A/B(lda/ldb)
    // NOTE: N = Col count
    status = cublasSetMatrix(M, N, sizeof(*arrayC), arrayC, M, deviceArrayC, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: Failed to move data to device.\n");
        printf("Freeing Memory and exiting\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        cudaFree(deviceArrayC)
;
        free(arrayA);
        free(arrayB);
        free(arrayC);
        cublasDestroy(handle);
        return -1;
    }
    
    #ifdef DEBUG
        printf("SUCCESS: Data transfered to device\n");
    #endif
    
    // Currently set to 1 and 0 respectively to test GEMM functionality
    float alphaScalar = 1.0f;
    float betaScalar = 0.0f;
    
    
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    
    #ifdef DEBUG
        printf("Performing GPU Operation\n");
    #endif
    
    // Marks current time for measuring performance
    auto start = chrono::high_resolution_clock::now();
    
    cublasSgemmEx(handle, 
                  transA, 
                  transB, 
                  M, 
                  N, 
                  N, 
                  &alphaScalar, 
                  deviceArrayA, 
                  CUDA_R_16F, 
                  M, 
                  deviceArrayB, 
                  CUDA_R_16F, 
                  M, 
                  &betaScalar, 
                  deviceArrayC, 
                  CUDA_R_16F, 
                  M);
    
    
    // Forces device to finish before proceeding (for timing)
    cudaDeviceSynchronize();

    
    // Marks current time for measuring performance
    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = finish - start;
    
    printf("HGEMM: %d x %d -- %fs\n",M,N,elapsed.count());
    
    
    /*

    cublasHgemm(handle, transA, transB, M, N, N, 
                &alphaScalar, deviceArrayA, M, 
                deviceArrayB, M, &betaScalar, 
                deviceArrayC, M);
    */
    
    status = cublasGetMatrix (M, N, sizeof(*arrayC), deviceArrayC, M, arrayC, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("Data Download Failed\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        cudaFree(deviceArrayC);
        free(arrayA);
        free(arrayB);    
        free(arrayC);
        cublasDestroy(handle);
        return -1;
    }
    
    #ifdef DEBUG
        printf("GPU Operation Completed\nFreeing Resources\n");
    #endif
    
    // Frees device pointers from cuda memory
    cudaFree(deviceArrayA);
    cudaFree(deviceArrayB);
    cudaFree(deviceArrayC);
    
    // Uninitializes the cuBLAS hanlder
    cublasDestroy(handle);

    // Frees host pointers from memory
    free(arrayA);
    free(arrayB);
    free(arrayC);
    
    return EXIT_SUCCESS;

}

int main() {
    
    for (int x = 0; x <= 30000; x += 5000) {
        if (x != 0) {
            hgemm_test(x,x);
        }
    }
    
    return EXIT_SUCCESS;
}
