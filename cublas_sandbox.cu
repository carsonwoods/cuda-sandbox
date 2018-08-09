#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 6
#define N 5
#define IDX2c(i,j,ld) (((j)*(ld))+(i))
