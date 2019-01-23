# -Wall:  Show Compilation Errors AND Warnings
# -std=c++17 Compile for C++ 17 Standard

all : add_cpu add_cuda add_cuda_block max cublas

add_cpu: add.cpp
	g++ -std=c++17 -Wall add.cpp -o add_cpu

add_cuda: vectorAdd.cu
	nvcc vectorAdd.cu -o add_cuda

max: max.cu
	nvcc max.cu -o max	

hgemm: hgemm_test.cu
	nvcc -std=c++11 hgemm_test.cu -lcublas -o hgemm_test_out

cublas: cublas_sample.cu cublas_sandbox.cu hgemm_test.cu
	nvcc -std=c++11 hgemm_test.cu -lcublas -o hgemm_test_out
	nvcc -std=c++11 cublas_sample.cu -lcublas -o cublas_sample
	nvcc -std=c++11 cublas_sandbox.cu -lcublas -o cublas

clean: 
	rm add_cpu add_cuda add_cuda_block max cublas cublas_sample hgemm_test_out
