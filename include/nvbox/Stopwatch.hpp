#ifndef STOPWATCH_HPP_
#define STOPWATCH_HPP_

#include <chrono>

namespace nvbox
{
    class Stopwatch
    {
    private:
        bool _isRunning;
        std::chrono::high_resolution_clock::time_point _stopPoint;
        std::chrono::high_resolution_clock::time_point _startPoint;

    public:
        void Start();
        void Stop();

        int64_t GetElapsedTime() const;

        static Stopwatch StartNew();

    private:
        std::chrono::high_resolution_clock::time_point GetNowTimePoint() const;
    };
}

#endif //! STOPWATCH_HPP_