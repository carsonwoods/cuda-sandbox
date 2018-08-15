#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <iostream>

#define INDEX(i,j,ld) (((j)*(ld))+(i))

using namespace std;

int runCublasSscal() {
    
    int M = 6; //Column Count
    int N = 5; //Row Count

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

int runCublasSgemm() {

    int M = 2;
    int N = 2;

    cudaError_t cudaErr;

    cublasStatus_t status;
    cublasHandle_t handle;

    int i, j; //Will be used to iterate through array

    float* arrayA = 0; //will store a copy of the array on host
    float* arrayB = 0;
    float* arrayC = 0;
    
    float* deviceArrayA; //will store a copy of the array on device
    float* deviceArrayB; 
    float* deviceArrayC;
    
    arrayA = (float *)malloc(M * N * sizeof(arrayA));
    arrayB = (float *)malloc(N * N * sizeof(arrayB));
    arrayC = (float *)malloc(N * N * sizeof(arrayC));


    //Ensures that host memory is allocated before proceeding
    if (!arrayA || !arrayB || !arrayC) {
        cout << "Host memory allocation failed\n" << endl;
        return -1;
    }

    //Iterates through hostArray and assigns value to all memory spaces
    for (j = 0; j < N; j++) {
        //first loop iterates through all rows
        for (i = 0; i < M; i++) {
            //second loop iterates through all columns within a row
            //Initializes array with values.
            arrayA[INDEX(i,j,M)] = (float)(i * M + j + 1);
            arrayB[INDEX(i,j,M)] = (float)(i * M + j + 2);
            arrayC[INDEX(i,j,M)] = 0;
        }
    }
    

    //iterates over array and prints result
    printf("arrayA:\n");
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", arrayA[INDEX(i,j,M)]);
        }
        printf ("\n");
    }
    
    //iterates over array and prints result
    printf("arrayB:\n");
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", arrayB[INDEX(i,j,M)]);
        }
        printf ("\n");
    }
    

    cudaErr = cudaMalloc((void**)&deviceArrayA, M*N*sizeof(*arrayA));
    if (cudaErr != cudaSuccess) {
        printf("Device memory allocation failed\n");
        return -1;
    }

    cudaErr = cudaMalloc((void**)&deviceArrayB, M*N*sizeof(*arrayB));
    if (cudaErr != cudaSuccess) {
        printf("Device memory allocation failed\n");
        return -1;
    }
    
    cudaErr = cudaMalloc((void**)&deviceArrayC, M*N*sizeof(*arrayC));
    if (cudaErr != cudaSuccess) {
        printf("Device memory allocation failed\n");
        return -1;
    }
    
    printf("Device memory allocated successfully\n");
    
    //initializes cublas API
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

    printf("cuBLAS API Initialized Successfully\n");
    
    //copies matrix from host to device
    status = cublasSetMatrix(M, N, sizeof(*arrayA), arrayA, M, deviceArrayA, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Data transfer to device failed\n");
        printf("Freeing Memory and exiting\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        cudaFree(deviceArrayC);
        cublasDestroy(handle);
        return -1;
    }
    
    //copies matrix from host to device
    status = cublasSetMatrix(M, N, sizeof(*arrayB), arrayB, M, deviceArrayB, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Data transfer to device failed\n");
        printf("Freeing Memory and exiting\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        cudaFree(deviceArrayC);
        cublasDestroy(handle);
        return -1;
    }
    
    //copies matrix from host to device
    status = cublasSetMatrix(M, N, sizeof(*arrayC), arrayC, M, deviceArrayC, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Data transfer to device failed\n");
        printf("Freeing Memory and exiting\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        cudaFree(deviceArrayC);
        cublasDestroy(handle);
        return -1;
    }
    
    printf("Data successfully uploaded to device\n");
    
    //Currently set to 1 because I just want to test GEMM functionality
    float alphaScalar = 1.0f;
    float betaScalar = 0.0f;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    
    printf("Performing GPU Operation\n");
    
    cublasSgemm(handle, transa, transb, M, N, 2, &alphaScalar, deviceArrayA, M,
                deviceArrayB, M, &betaScalar, deviceArrayC, M);
    
    status = cublasGetMatrix (M, N, sizeof(*arrayC), deviceArrayC, M, arrayC, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("Data Download Failed\n");
        cudaFree(deviceArrayA);
        cudaFree(deviceArrayB);
        cudaFree(deviceArrayC);
        cublasDestroy(handle);
        return -1;
    }
    
    printf("Data Downloaded Successfully\n");
    
    //Frees device pointers from cuda memory
    cudaFree(deviceArrayA);
    cudaFree(deviceArrayB);
    cudaFree(deviceArrayC);
    
    //Uninitializes the cuBLAS hanlder
    cublasDestroy(handle);

    printf("Result:\n");
    
    //Iterates over array and prints result
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", arrayC[INDEX(i,j,M)]);
        }
        printf ("\n");
    }

    //Frees host pointers from memory
    free(arrayA);
    free(arrayB);
    free(arrayC);

    return 0;
}


int main() {
    
    //runCublasSscal();
    runCublasSgemm();

    return 0;
}
