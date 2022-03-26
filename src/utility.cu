#include <cstdio>
#include <string>

#include "nvbox/utility.cuh"

void nvbox::describeCuda()
{
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

void nvbox::describeCudaDevices()
{
    int deviceCount = 0;
    handleCudaError(cudaGetDeviceCount(&deviceCount));

    printf("CUDA devices:\t\t%d\n", deviceCount);

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
    {
        printf("===\nDevice:\t%d\n", deviceId);

        int attributeValue;

        handleCudaError(cudaDeviceGetAttribute(&attributeValue, cudaDeviceAttr::cudaDevAttrMaxThreadsPerBlock, deviceId));
        printf("MaxThreadsPerBlock:\t%d\n", attributeValue);

        handleCudaError(cudaDeviceGetAttribute(&attributeValue, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, deviceId));
        printf("MaxThreadsPerMultiProcessor:\t%d\n", attributeValue);

        handleCudaError(cudaDeviceGetAttribute(&attributeValue, cudaDeviceAttr::cudaDevAttrClockRate, deviceId));
        printf("ClockRate:\t%d\n", attributeValue);

        handleCudaError(cudaDeviceGetAttribute(&attributeValue, cudaDeviceAttr::cudaDevAttrMemoryClockRate, deviceId));
        printf("MemoryClockRate:\t%d\n", attributeValue);

        cudaDeviceProp deviceProp;
        handleCudaError(cudaGetDeviceProperties(&deviceProp, deviceId));
    }
}

void nvbox::handleCudaError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        int errorCode = static_cast<int>(error);
        const char *errorString = cudaGetErrorString(error);

        printf("Cannot get device count.\n[ERR %d] => %s\n", errorCode, errorString);

        exit(EXIT_FAILURE);
    }
}