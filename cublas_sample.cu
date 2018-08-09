//Example 2. Application Using C and CUBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>

#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    
    //cublasSscal scales the vector &m by the scalar &alpha and overwrites it with the result
    //PARAMETER BREAKDOWN:  
    //handle: cublasHandle to call the cublas API, n-p: number of elements in the vector &m. &alpha: scalar used to scale &m.
    //&m: device copy of vector with n-p elements. ldm: stride between consecutive elements of x.
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}

int main (void){
   
    //checks to ensure there are no cudaErrors
    cudaError_t cudaStat;    

    //cublas specific objects
    //status handles the status of cublas. similar to cudaError_t
    //handle is the invocation object of cublas api
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    int i, j;
    float* devPtrA;
    
    //Creates pointer a then allocates it as CPU memory.
    float* a = 0;
    a = (float *)malloc (M * N * sizeof (*a));
    //ensures that a was allocated on the host
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }


    //iterates through memory space of a and assigns each index a value
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }

    //allocates GPU memory for device pointer A and ensures it worked
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }


    //initializes cublas through &handle and ensures success
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    //Basically acts as cudaMemcpy to device from host. Copies a tile of (M rows)x(N cols). 
    //M: Row count. N: Col count. sizeof(*a): elemSize. a: source matrix
    //M(second instance): leading dimension of A. devPtrA: destination matrix. M(third instance): leading dimension of devPtrA.
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //calls modify: inline function declared above
    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);

    //Basically acts as cudaMemcpy to host from device. Copies a tile of (M rows)x(N cols). 
    //M: Row count. N: Col count. sizeof(*a): elemSize. a: source matrix
    //M(second instance): leading dimension of A. devPtrA: destination matrix. M(third instance): leading dimension of devPtrA.
    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //Frees device pointer from cuda memory
    cudaFree (devPtrA);

    //uninitialize the cublas hanlder
    cublasDestroy(handle);

    //iterates over array and prints result
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }

    //free host pointer a from memory
    free(a);

    //return with successful message
    return EXIT_SUCCESS;
}
