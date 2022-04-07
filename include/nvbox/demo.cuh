#ifndef DEMO_CUH_
#define DEMO_CUH_

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace nvbox
{
    void RunAddDemo();

    __global__ void VecAdd(float *a, float *b, float *c, int n);
}

#endif // DEMO_CUH_