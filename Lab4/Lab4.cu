/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 11/04/2923
Description:
Lab4: Use three functions that internally use different memory models 
      to perform the calculations (random walking in 2D).
*/
#include <stdio.h> 
#include <math.h>
#include <curand_kernel.h>
#include <ctime>
#include <chrono>
#include <string>
#include <iostream>

using namespace std;
using namespace chrono;

// GPU-function, input Out-array, a (x-direction array), b (y-direction array), random-seed, S and W
__global__ void vector_distance(float* out, float* a, float* b, unsigned int seed, int m, int n)
{
    // calculate tid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // cuda_random
    curandState state;
    curand_init(seed, tid, 0, &state);

    // local variables to save the position
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    // Make sure the current thread is correct
    if (tid < n)
    {
        // for each walker, walk S steps
        for (int i = 0; i < m; ++i)
        {
            // get random distances for 1 step in X and Y direction
            float x = curand_uniform(&state) * 1.0 - 0.5;
            float y = curand_uniform(&state) * 1.0 - 0.5;
            sum_x += x;
            sum_y += y;
        }
        a[tid] += sum_x;
        b[tid] += sum_y;
        // get square of the distance
        out[tid] = (a[tid] * a[tid]) + (b[tid] * b[tid]);
    }
}

int main(int argc, char* argv[])
{
    // Default Walkers and Steps
    int def_argc = 5;
    int W = 100;
    int S = 100;
    // Error parameters in command.
    if (argc < def_argc)
    {
        cout << "Error parameters! Please try again!" << endl;
        return;
    }
    else
    {
        // get the inputs of W and S
        W = atoi(argv[2]);
        S = atoi(argv[4]);
        if (W < 1 || S < 1)
        {
            cout << "Error parameters! Please try again!" << endl;
            return;
        }
    }

    // Get CUDA information
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    // check whether the Device is Integrated.
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        //printf("Device name: %s\n", prop.name);
        if (prop.integrated)
        {
            //cout << prop.integrated <<endl;
            cout << "Integrated GPU! Please try again!\n";
            return;
        }
    }

    // initial
    int block_size = 256;
    int grid_size = ((W + block_size) / block_size);

    float sum_dst = 0.0f;
    const unsigned int Bytes = sizeof(float) * W;
    high_resolution_clock::time_point startTime, endTime;

    //First Method;
    cout << "Normal CUDA memory Allocation:\n";
    float* x, * y, * dst;
    float* d_x, * d_y, * d_dst;

    // Allocate host memory
    x = (float*)malloc(Bytes);
    y = (float*)malloc(Bytes);
    dst = (float*)malloc(Bytes);
    memset(x, 0, sizeof(x));
    memset(y, 0, sizeof(y));

    // Begin the timer for normal function
    startTime = std::chrono::high_resolution_clock::now();
    // Allocate device memory 
    cudaMalloc((void**)&d_x, Bytes);
    cudaMalloc((void**)&d_y, Bytes);
    cudaMalloc((void**)&d_dst, Bytes);

    // Transfer data from host to device memory
    cudaMemcpy(d_x, x, Bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, Bytes, cudaMemcpyHostToDevice);

    // Executing kernel 
    vector_distance<<<grid_size, block_size >>>(d_dst, d_x, d_y, time(NULL), S, W);

    // Transfer data back to host memory
    cudaMemcpy(dst, d_dst, Bytes, cudaMemcpyDeviceToHost);
    //cudaMemcpy(x, d_x, sizeof(float) * W, cudaMemcpyDeviceToHost);
    //cudaMemcpy(y, d_y, sizeof(float) * W, cudaMemcpyDeviceToHost);

    // End the timer for normal function
    endTime = std::chrono::high_resolution_clock::now();

    // Calculate the average distance
    for (int i = 0; i < W; i++)
    {
        sum_dst += sqrt(dst[i]);
        //cout << dst[i] << " x:" << x[i] << " y:" << y[i] << endl;
    }
    sum_dst = sum_dst / W;
    auto duration = duration_cast<microseconds>(endTime - startTime);
    cout << "    Time to calculate(microsec):" << duration.count() << endl;
    cout << "    Average distance from origin: " << sum_dst << endl;

    // Deallocate device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dst);
    // Deallocate host memory
    free(x);
    free(y);
    free(dst);




    //Second Method;
    float* h_xPinned, * h_yPinned, * h_dPinned;
    // device array
    float* d_xPinned, * d_yPinned, * d_dPinned;
    sum_dst = 0.0f;
    cout << "Pinned CUDA memory Allocation:\n";

    // Begin the timer for Pinned function
    startTime = std::chrono::high_resolution_clock::now();
    cudaMallocHost((void**)&h_xPinned, Bytes);  // host pinned x_array
    cudaMallocHost((void**)&h_yPinned, Bytes);  // host pinned y_array
    cudaMallocHost((void**)&h_dPinned, Bytes);  // host pinned distance_array
    cudaMalloc((void**)&d_xPinned, Bytes);      // device x_array
    cudaMalloc((void**)&d_yPinned, Bytes);      // device y_array
    cudaMalloc((void**)&d_dPinned, Bytes);      // device distance_array
    memset(h_xPinned, 0, sizeof(h_xPinned));    // Initial h_x
    memset(h_yPinned, 0, sizeof(h_xPinned));    // Initial h_y

    // Transfer data from host to device memory
    cudaMemcpy(d_xPinned, h_xPinned, Bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_yPinned, h_yPinned, Bytes, cudaMemcpyHostToDevice);
    // Executing kernel 
    vector_distance<<<grid_size, block_size >>>(d_dPinned, d_xPinned, d_yPinned, time(NULL), S, W);
    // Transfer data back to host memory
    cudaMemcpy(h_dPinned, d_dPinned, Bytes, cudaMemcpyDeviceToHost);

    // End the timer for Pinned function
    endTime = std::chrono::high_resolution_clock::now();
    auto duration_Pinned = duration_cast<microseconds>(endTime - startTime);

    for (int i = 0; i < W; i++)
    {
        sum_dst += sqrt(h_dPinned[i]);
    }
    sum_dst = sum_dst / W;
    cout << "    Time to calculate(microsec):" << duration_Pinned.count() << endl;
    cout << "    Average distance from origin: " << sum_dst << endl;
    
    // Deallocate device memory
    cudaFree(d_xPinned);
    cudaFree(d_yPinned);
    cudaFree(d_dPinned);
    // Deallocate host memory
    cudaFreeHost(h_xPinned);
    cudaFreeHost(h_yPinned);
    cudaFreeHost(h_dPinned);



    //Third Method;
    float* xManaged, * yManaged, * dManaged;
    sum_dst = 0.0f;
    cout << "Managed CUDA memory Allocation:\n";

    // Begin the timer for Managed function
    startTime = std::chrono::high_resolution_clock::now();
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&xManaged, Bytes);    //Managed Memory for x_direction
    cudaMallocManaged(&yManaged, Bytes);    //Managed Memory for y_direction
    cudaMallocManaged(&dManaged, Bytes);    //Managed Memory for distance
    memset(xManaged, 0, sizeof(xManaged));  //Initial x_array
    memset(yManaged, 0, sizeof(yManaged));  //Initial y_array

    // Executing kernel 
    vector_distance<<<grid_size, block_size >>>(dManaged, xManaged, yManaged, time(NULL), S, W);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // End the timer for Managed function
    endTime = std::chrono::high_resolution_clock::now();
    auto duration_Managed = duration_cast<microseconds>(endTime - startTime);
    for (int i = 0; i < W; i++)
    {
        sum_dst += sqrt(dManaged[i]);
    }
    sum_dst = sum_dst / W;
    cout << "    Time to calculate(microsec):" << duration_Managed.count() << endl;
    cout << "    Average distance from origin: " << sum_dst << endl;
    cout << "Bye" << endl;
    // Free memory
    cudaFree(xManaged);
    cudaFree(yManaged);
    cudaFree(dManaged);
}