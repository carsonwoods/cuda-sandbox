# -Wall:  Show Compilation Errors AND Warnings
# -std=c++17 Compile for C++ 17 Standard

all : add_cpu add_cuda add_cuda_block

add_cpu: add.cpp
	g++ -std=c++17 -Wall add.cpp -o add_cpu

add_cuda: add.cu
	nvcc add.cu -o add_cuda

add_cuda_block: add_blocks.cu
	nvcc -arch=compute_61  add_blocks.cu -o add_cuda_block

max: max.cu
	nvcc -arch=compute_61 max.cu -o max

clean: 
	rm add_cpu add_cuda add_cuda_block max
