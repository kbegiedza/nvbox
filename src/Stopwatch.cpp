#include "nvbox/Stopwatch.hpp"

void nvbox::Stopwatch::Start()
{
    _isRunning = true;
    _startPoint = GetNowTimePoint();
}

void nvbox::Stopwatch::Stop()
{
    _stopPoint = GetNowTimePoint();
    _isRunning = false;
}

int64_t nvbox::Stopwatch::GetElapsedTime() const
{
    auto endPoint = _isRunning ? GetNowTimePoint() : _stopPoint;

    return std::chrono::duration_cast<std::chrono::milliseconds>(endPoint - _startPoint).count();
}

nvbox::Stopwatch nvbox::Stopwatch::StartNew()
{
    nvbox::Stopwatch stopwatch;

    stopwatch.Start();

    return stopwatch;
}

std::chrono::high_resolution_clock::time_point nvbox::Stopwatch::GetNowTimePoint() const
{
    return std::chrono::high_resolution_clock::now();
}