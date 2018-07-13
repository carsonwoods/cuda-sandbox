# -Wall:  Show Compilation Errors AND Warnings
# -std=c++17 Compile for C++ 17 Standard

all : TestAdd

TestAdd: add.cpp
	g++ -std=c++17 -Wall add.cpp -o TestAdd

add_cuda: add.cu
	nvcc add.cu -o add_cuda

add_cuda_block: add_block.cu
	nvcc add_block.cu -o add_cuda_block
