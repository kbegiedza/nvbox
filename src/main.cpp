#include <cstdio>

#include "nvbox/utility.cuh"

int main()
{
    nvbox::describeCuda();
    nvbox::describeCudaDevices();

    return 0;
}