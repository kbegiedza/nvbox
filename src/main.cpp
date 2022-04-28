#include <chrono>
#include <thread>
#include <iostream>

#include <nvml.h>
#include <signal.h>

#include "nvbox/Stopwatch.hpp"
#include "nvbox/utility.cuh"

#include <argparse/argparse.hpp>

argparse::ArgumentParser create_parser()
{
    argparse::ArgumentParser parser("nvbox");

    parser.add_description("Toolbox for cuda");

    parser.add_argument("-r", "--refresh")
        .help("Refresh internal in milliseconds [ms]\nDefault value = 1000 [ms]")
        .scan<'i', int>()
        .default_value(1000);

    return parser;
}

void handle_sigint(sig_atomic_t s)
{
    nvmlShutdown();

    std::exit(0);
}

void attach_to_signals()
{
    signal(SIGINT, handle_sigint);
}

// TODO: use cli framework
void clear_console()
{
#if defined _WIN32
    system("cls");
#elif defined(__LINUX__) || defined(__gnu_linux__) || defined(__linux__)
    std::cout << "\x1B[2J\x1B[H";
#elif defined(__APPLE__)
    system("clear");
#endif
}

int main(int argc, char *argv[])
{
    auto parser = create_parser();

    try
    {
        parser.parse_args(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;

        std::exit(1);
    }

    attach_to_signals();

    auto refreshTime = parser.get<int>("--refresh");

    std::chrono::milliseconds sleepTime(refreshTime);

    nvmlInit_v2();

    uint32_t deviceCount = 0;

    nvmlReturn_t nvmlOpStatus;
    nvmlOpStatus = nvmlDeviceGetCount_v2(&deviceCount);

    if (nvmlOpStatus != nvmlReturn_t::NVML_SUCCESS)
    {
        std::cerr << "Unable to query device count\n"
                  << nvmlOpStatus
                  << std::endl;

        std::exit(1);
    }

    if (deviceCount <= 0)
    {
        std::cerr << "Unable to find any device"
                  << std::endl;

        std::exit(1);
    }

    std::vector<nvmlDevice_t> devices;

    for (uint32_t deviceId = 0; deviceId < deviceCount; ++deviceId)
    {
        nvmlDevice_t currentDevice;
        nvmlOpStatus = nvmlDeviceGetHandleByIndex_v2(deviceId, &currentDevice);
        if (nvmlOpStatus != nvmlReturn_t::NVML_SUCCESS)
        {
            std::cerr << "Unable to get handle to device"
                      << deviceId
                      << std::endl
                      << nvmlOpStatus
                      << std::endl;

            break;
        }

        devices.push_back(std::move(currentDevice));
    }

    while (true)
    {
        clear_console();

        for (auto &&device : devices)
        {
            uint32_t temp, temp_count;

            char *deviceUUIDPtr = new char[NVML_DEVICE_UUID_V2_BUFFER_SIZE];

            nvmlDeviceGetUUID(device, deviceUUIDPtr, NVML_DEVICE_UUID_V2_BUFFER_SIZE);

            nvmlDeviceGetTemperature(device, nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU, &temp);

            std::cout << "Device: " << deviceUUIDPtr
                      << "\n=========================\n"
                      << "Temp:\t" << temp << "°C"
                      << std::endl;

            delete[] deviceUUIDPtr;
            deviceUUIDPtr = nullptr;
        }

        std::this_thread::sleep_for(sleepTime);
    }

    nvmlShutdown();

    std::cout << "Done" << std::endl;

    return 0;
}