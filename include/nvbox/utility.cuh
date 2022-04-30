#ifndef NVBOX_UTILITY_CUH_
#define NVBOX_UTILITY_CUH_

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace nvbox
{
    void describeCuda();

    void describeCudaDevices();

    void describeTemperatures();

    void handleCudaError(cudaError_t error);
}

#endif //! NVBOX_UTILITY_CUH_