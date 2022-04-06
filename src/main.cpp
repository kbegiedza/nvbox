#include <iostream>

#include "nvbox/Stopwatch.hpp"
#include "nvbox/utility.cuh"

int main()
{
    auto sw = nvbox::Stopwatch::StartNew();

    nvbox::describeCuda();
    nvbox::describeCudaDevices();

    sw.Stop();

    std::cout << std::endl
              << "Finished in: "
              << sw.GetElapsedTime()
              << " [ms]"
              << std::endl;

    return 0;
}