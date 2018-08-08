# -Wall:  Show Compilation Errors AND Warnings
# -std=c++17 Compile for C++ 17 Standard

all : add_cpu add_cuda add_cuda_block max cublas

add_cpu: add.cpp
	g++ -std=c++17 -Wall add.cpp -o add_cpu

add_cuda: add.cu
	nvcc add.cu -o add_cuda

add_cuda_block: add_blocks.cu
	nvcc -arch=compute_61  add_blocks.cu -o add_cuda_block

max: max.cu
	nvcc max.cu -o max

cublas: cublas_test.cu
	nvcc cublas_test.cu -lcublas -o cublas

clean: 
	rm add_cpu add_cuda add_cuda_block max cublas
