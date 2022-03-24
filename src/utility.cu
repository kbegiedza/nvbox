#ifndef NVBOX_UTILITY_HPP_
#define NVBOX_UTILITY_HPP_

#include <cstdio>
#include <string>
#include <nvbox/utility.hpp>
#include <cuda_runtime.h>

void nvbox::describeCuda()
{
    printf("Quering CUDA devices...\n");

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        printf("Cannot get device count.\n[ERR %d] => %s\n",
               static_cast<int>(error),
               cudaGetErrorString(error));

        return;
    }

    printf("CUDA devices:\t\t%d\n", deviceCount);

    int driverVersion, runtimeVersion;

    cudaDriverGetVersion(&driverVersion);
    cudaDriverGetVersion(&runtimeVersion);

    auto getCudaVersionString = [&](int version) -> std::string
    {
        return std::to_string(version / 1000) + "." + std::to_string((version % 100) / 10);
    };

    printf("CUDA Driver Version:\t%s\nRuntime Version:\t%s\n",
           getCudaVersionString(driverVersion).c_str(),
           getCudaVersionString(runtimeVersion).c_str());
}
#endif