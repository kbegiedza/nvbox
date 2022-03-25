#include <cstdio>

#include <nvbox/utility.hpp>

int main()
{
    nvbox::describeCuda();
    nvbox::describeCudaDevices();

    return 0;
}