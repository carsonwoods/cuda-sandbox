#include <iostream>
#include <math.h>

using namespace std;



void add(int n, float *x, float *y) {
    int z = 0;
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
        z++;
    }
}

int main() {

    //For my own sanity lets explain this.
    //1<<20 is a notation that in this context represents
    //a bitshift. That means that you have the bit 1 and then you shift it to the
    //(in this case) left by 20 spaces and fill the empty space with zeros.
    int N = 1<<25; // 1M elements

    //creates memory space (pointer) to two arrays of 1 million floats
    float *x = new float[N];
    float *y = new float[N];

    //initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    //Run kernel on 1M elements on the CPU
    add(N, x, y);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}
