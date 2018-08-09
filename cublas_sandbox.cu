#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>


#define M 6 //Column Count
#define N 5 //Row Count
#define INDEX(i,j,ld) (((j)*(ld))+(i))

using namespace std;

int main() {
    
    cudaError_t cudaErr;

    cublasStatus_t status;
    cublasHandle_t handle;

    int i, j; //Will be used to iterate through array

    float* hostArray = 0; //will store a copy of the array on host
    float* deviceArray; //will store a copy of the array on device

    hostArray = (float *)malloc(M * N * sizeof(hostArray));

    if (!hostArray) {
        cout << "Host memory allocation failed" << endl;
        return -1;
    }

    for (j = 0; j < N; j++) {
        //first loop iterates through all rows
        for (i = 0; i < M; i++) {
            //second loop iterates through all columns within a row
            //Initializes array with values.
            hostArray[INDEX(i,j,M)] = (float)(i * M + j + 1);           
        }
    }

    cudaErr = cudaMalloc((void**)&deviceArray, M*N*sizeof(*hostArray));
    if (cudaErr != cudaSuccess) {
        printf("Device memory allocation failed");
        return -1;
    }

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API Initialization Failed\n");
        cudaFree(deviceArray);
        free(hostArray);
        return -1;
    } else {
         printf("cuBLAS API Initialized Successfully\n");
    }

    status = cublasSetMatrix(M, N, sizeof(*hostArray), hostArray, M, deviceArray, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Data transfer to device failed\n");
        printf("Freeing Memory and exiting\n");
        cudaFree(deviceArray);
        cublasDestroy(handle);
        return -1;
    } else {
        printf("Data successfully uploaded to device\n");
    }
    
    float alphaScalar = 16.0f;
    float betaScalar = 12.0f;

    cublasSscal(handle, N-1, &alphaScalar, &deviceArray[INDEX(1,2,M)], M);
    cublasSscal(handle, M-1, &betaScalar, &deviceArray[INDEX(1,2,M)], 1);

    status = cublasGetMatrix (M, N, sizeof(*hostArray), deviceArray, M, hostArray, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("Data Download Failed");
        cudaFree (deviceArray);
        cublasDestroy(handle);
        return -1;
    }

    //Frees device pointer from cuda memory
    cudaFree(deviceArray);

    //uninitialize the cublas hanlder
    cublasDestroy(handle);

    //iterates over array and prints result
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", hostArray[INDEX(i,j,M)]);
        }
        printf ("\n");
    }

    //free host pointer a from memory
    free(hostArray);

    return 0;





}
