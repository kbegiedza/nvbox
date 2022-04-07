#include "nvbox/demo.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace nvbox
{
    __global__ void VecAdd(float *a, float *b, float *c, int n)
    {
        int i = blockDim.x * blockDim.x + threadIdx.x;

        if (i < n)
        {
            *(c + i) = *(a + i) + *(b + i);
        }
    }

    void RunAddDemo()
    {
        int N = 100;
        size_t size = N * sizeof(float);

        float *hostA = (float *)malloc(size);
        float *hostB = (float *)malloc(size);
        float *hostC = (float *)malloc(size);

        float *deviceA;
        float *deviceB;
        float *deviceC;

        cudaMalloc(&deviceA, size);
        cudaMalloc(&deviceB, size);
        cudaMalloc(&deviceC, size);

        cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        nvbox::VecAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, N);

        cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost);

        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);

        free(hostA);
        free(hostB);
        free(hostC);
    }
}