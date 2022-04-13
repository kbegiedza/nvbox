#include <chrono>
#include <thread>
#include <iostream>

#include <signal.h>

#include "nvbox/Stopwatch.hpp"
#include "nvbox/utility.cuh"
#include "nvbox/demo.cuh"

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

    while (true)
    {
        clear_console();

        std::cout << "1" << std::endl;

        std::this_thread::sleep_for(sleepTime);
    }

    std::cout << "Done" << std::endl;

    return 0;
}