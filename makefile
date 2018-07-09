# -Wall:  Show Compilation Errors AND Warnings
# -std=c++17 Compile for C++ 17 Standard

all : TestAdd

TestAdd: add.cpp
	g++ -std=c++17 -Wall add.cpp -o TestAdd
