#include <iostream>
#include <cmath>
#include <curand_kernel.h>
#include <string>
#include <ctime>

using namespace std;
__global__ void simulateRandomWalk(float *positions,int numSteps, int seed) {
    /* random walk simulation. return distance from origin after random walk as a float array */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, tid, 0, &state);

    float x = 0.0f;
    float y = 0.0f;

    for (int step = 0; step < numSteps; step++) {

        float random_direction = curand_uniform(&state) * 4.0;
        if (random_direction < 1.0) {
            x += 1.0;
        } else if (random_direction < 2.0) {
            x -= 1.0;
        } else if (random_direction < 3.0) {
            y += 1.0;
        } else {
            y -= 1.0;
        }
    }

    positions[tid] = sqrt(x * x + y * y);
}

unsigned long long int normalMemoryWalk(float *&positions, int numWalkers, int numSteps) {
    float *d_positions;

    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    cudaEventRecord(startEvent);

    cudaMalloc((void **)&d_positions, numWalkers * sizeof(float));

    int seed = time(NULL);


    simulateRandomWalk<<<numWalkers, 1>>>(d_positions, numSteps, seed);

    cudaDeviceSynchronize();

    cudaMemcpy(positions, d_positions, numWalkers * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_positions);

    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, endEvent);


    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    return static_cast<unsigned long long int>(milliseconds * 1000);
}

unsigned long long int pinnedMemoryWalk(float *&positions, int numWalkers,int numSteps) {
    float *d_positions;

    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    cudaEventRecord(startEvent);

    cudaMallocHost((void **)&d_positions, numWalkers * sizeof(float));

    int seed = time(NULL);


    simulateRandomWalk<<<numWalkers, 1>>>(d_positions, numSteps, seed);

    cudaDeviceSynchronize();

    cudaMemcpy(positions, d_positions, numWalkers * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_positions);

    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, endEvent);


    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    return static_cast<unsigned long long int>(milliseconds * 1000);
}
unsigned long long int managedMemoryWalk(float *&positions, int numWalkers,int numSteps) {
    float *d_positions;

    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    cudaEventRecord(startEvent);

    cudaMallocManaged((void **)&d_positions, numWalkers * sizeof(float));

    int seed = time(NULL);


    simulateRandomWalk<<<numWalkers, 1>>>(d_positions, numSteps, seed);

    cudaDeviceSynchronize();

    cudaMemcpy(positions, d_positions, numWalkers * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_positions);

    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, endEvent);


    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    return static_cast<unsigned long long int>(milliseconds * 1000);
}
int main(int argc, char **argv) {
    int numWalkers, numSteps;
    if (argc != 5) {
        numWalkers=1000;
        numSteps=100000;
    }


    for (int i = 1; i < argc; i += 2) {
        if (string(argv[i]) == "-W")
            numWalkers = stoi(argv[i + 1]);
        else if (string(argv[i]) == "-I")
            numSteps = stoi(argv[i + 1]);
    }

    float *positions;
    positions = new float[numWalkers];

    unsigned long long timeNormal, timePinned, timeManaged;

    timeNormal=normalMemoryWalk(positions,numWalkers,numSteps);
    timePinned = pinnedMemoryWalk(positions,numWalkers,numSteps);
    timeManaged= managedMemoryWalk(positions,numWalkers,numSteps);


    float averageDistance = 0.0f;
    for (int i = 0; i < numWalkers; i++) {
        averageDistance += positions[i];
    }
    averageDistance /= numWalkers;

    cout << "Normal CUDA memory Allocation:" << endl;
    cout << "    Time to calculate(microsec): " << timeNormal << endl;
    cout << "    Average distance from origin: " << averageDistance << endl;

    cout << "Pinned CUDA memory Allocation:" << endl;
    cout << "    Time to calculate(microsec): " << timePinned << endl;
    cout << "    Average distance from origin: " << averageDistance << endl;

    cout << "Managed CUDA memory Allocation:" << endl;
    cout << "    Time to calculate(microsec): " << timeManaged << endl;
    cout << "    Average distance from origin: " << averageDistance << endl;

    cout << "Bye" << endl;


    delete[] positions;


    return 0;
}