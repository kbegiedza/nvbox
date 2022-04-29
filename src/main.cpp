#include <chrono>
#include <thread>
#include <iostream>

#include <nvml.h>
#include <signal.h>

#include "nvbox/Stopwatch.hpp"
#include "nvbox/utility.cuh"

#include "nvbox/DeviceStatusModel.hpp"
#include "nvbox/DeviceStatusService.hpp"
#include "nvbox/DeviceStatusView.hpp"

#include <argparse/argparse.hpp>

bool shouldExecute = true;

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

    shouldExecute = false;
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
    attach_to_signals();

    auto parser = create_parser();

    try
    {
        parser.parse_args(argc, argv);
    }
    catch (const std::exception &exception)
    {
        std::cerr << exception.what() << std::endl;

        std::exit(1);
    }

    auto refreshTime = parser.get<int>("--refresh");

    std::chrono::milliseconds sleepTime(refreshTime);

    nvmlInit_v2();

    // init device status
    auto model = std::make_shared<nvbox::DeviceStatusModel>();
    auto service = std::make_unique<nvbox::DeviceStatusService>(model);
    auto view = std::make_unique<nvbox::DeviceStatusView>(model);

    while (shouldExecute)
    {
        clear_console();

        service->Update();
        view->Render();

        shouldExecute = false;
        std::this_thread::sleep_for(sleepTime);
    }

    nvmlShutdown();

    clear_console();
    std::cout << "Done" << std::endl;

    return 0;
}