#ifndef NVBOX_UTILITY_HPP_
#define NVBOX_UTILITY_HPP_

#include <cuda_runtime.h>

namespace nvbox
{
    void describeCuda();

    void describeCudaDevices();

    void handleCudaError(cudaError_t error);
}

#endif //! NVBOX_UTILITY_HPP_