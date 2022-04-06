#ifndef NVBOX_UTILITY_CUH_
#define NVBOX_UTILITY_CUH_

#include <cuda_runtime.h>

namespace nvbox
{
    void describeCuda();

    void describeCudaDevices();

    void handleCudaError(cudaError_t error);
}

#endif //! NVBOX_UTILITY_CUH_