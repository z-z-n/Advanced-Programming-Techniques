/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 11/07/2923
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

// GPU-function, input Out-array, random-seed, S and W
__global__ void vector_distance(float* out, unsigned int seed, int m, int n)
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
        // get square of the distance
        out[tid] = (sum_x * sum_x) + (sum_y * sum_y);
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

    //First Method -- warm up;
    cout << "Normal CUDA memory Allocation:\n";
    float* dst;
    float* d_dst;

    // Allocate host memory
    dst = (float*)malloc(Bytes);

    // Begin the timer for normal function
    startTime = std::chrono::high_resolution_clock::now();
    // Allocate device memory 
    cudaMalloc((void**)&d_dst, Bytes);

    // Executing kernel
    vector_distance<<<grid_size, block_size >>>(d_dst, time(NULL), S, W);

    // Transfer data back to host memory
    cudaMemcpy(dst, d_dst, Bytes, cudaMemcpyDeviceToHost);

    // End the timer for normal function
    endTime = std::chrono::high_resolution_clock::now();

    // Deallocate device memory
    cudaFree(d_dst);
    // Deallocate host memory
    free(dst);


    //First Method
    // Begin the timer for normal function
    startTime = std::chrono::high_resolution_clock::now();
     // Allocate host memory
    dst = (float*)malloc(Bytes);

    // Allocate device memory 
    cudaMalloc((void**)&d_dst, Bytes);

    // Executing kernel
    vector_distance<<<grid_size, block_size >>>(d_dst, time(NULL), S, W);

    // Transfer data back to host memory
    cudaMemcpy(dst, d_dst, Bytes, cudaMemcpyDeviceToHost);

    // Calculate the average distance
    for (int i = 0; i < W; i++)
    {
        sum_dst += sqrt(dst[i]);
        //cout << dst[i] << " x:" << x[i] << " y:" << y[i] << endl;
    }
    sum_dst = sum_dst / W;

    // Deallocate device memory
    cudaFree(d_dst);
    // Deallocate host memory
    free(dst);

    // End the timer for normal function
    endTime = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(endTime - startTime);
    cout << "    Time to calculate(microsec):" << duration.count() << endl;
    cout << "    Average distance from origin: " << sum_dst << endl;

    


    //Second Method;
    float* h_dPinned;
    // device array
    float* d_dPinned;
    sum_dst = 0.0f;
    cout << "Pinned CUDA memory Allocation:\n";

    // Begin the timer for Pinned function
    startTime = std::chrono::high_resolution_clock::now();
    cudaMallocHost((void**)&h_dPinned, Bytes);  // host pinned distance_array
    cudaMalloc((void**)&d_dPinned, Bytes);      // device distance_array

    // Executing kernel 
    vector_distance<<<grid_size, block_size >>>(d_dPinned, time(NULL), S, W);
    // Transfer data back to host memory
    cudaMemcpy(h_dPinned, d_dPinned, Bytes, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < W; i++)
    {
        sum_dst += sqrt(h_dPinned[i]);
    }
    sum_dst = sum_dst / W;

    // Deallocate device memory
    cudaFree(d_dPinned);
    // Deallocate host memory
    cudaFreeHost(h_dPinned);

    // End the timer for Pinned function
    endTime = std::chrono::high_resolution_clock::now();
    auto duration_Pinned = duration_cast<microseconds>(endTime - startTime);
    cout << "    Time to calculate(microsec):" << duration_Pinned.count() << endl;
    cout << "    Average distance from origin: " << sum_dst << endl;
    



    //Third Method;
    float* dManaged;
    sum_dst = 0.0f;
    cout << "Managed CUDA memory Allocation:\n";

    // Begin the timer for Managed function
    startTime = std::chrono::high_resolution_clock::now();
    // Allocate Unified Memory-accessible from CPU or GPU
    cudaMallocManaged(&dManaged, Bytes);    //Managed Memory for distance

    // Executing kernel 
    vector_distance<<<grid_size, block_size >>>(dManaged, time(NULL), S, W);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    for (int i = 0; i < W; i++)
    {
        sum_dst += sqrt(dManaged[i]);
    }
    sum_dst = sum_dst / W;

    // Free memory
    cudaFree(dManaged);

    // End the timer for Managed function
    endTime = std::chrono::high_resolution_clock::now();
    auto duration_Managed = duration_cast<microseconds>(endTime - startTime);
    cout << "    Time to calculate(microsec):" << duration_Managed.count() << endl;
    cout << "    Average distance from origin: " << sum_dst << endl;
    cout << "Bye" << endl;
}