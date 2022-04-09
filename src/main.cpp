#include <iostream>

#include "nvbox/Stopwatch.hpp"
#include "nvbox/utility.cuh"
#include "nvbox/demo.cuh"

#include <argparse/argparse.hpp>

argparse::ArgumentParser create_parser()
{
    argparse::ArgumentParser parser("nvbox");

    parser.add_description("Toolbox for cuda");

    parser.add_argument("--demo")
        .help("Run simple demo")
        .default_value(false)
        .implicit_value(true);

    return parser;
}

void RunDemo()
{
    auto sw = nvbox::Stopwatch::StartNew();

    nvbox::describeCuda();
    nvbox::describeCudaDevices();

    nvbox::RunAddDemo();

    sw.Stop();

    std::cout << std::endl
              << "Finished in: "
              << sw.GetElapsedTime()
              << " [ms]"
              << std::endl;
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

    if (parser["--demo"] == true)
    {
        RunDemo();
    }

    return 0;
}