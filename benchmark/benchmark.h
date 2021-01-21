#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <chrono>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <fstream>
#include <vector>

#include <fmt/core.h>

namespace asc::pad_ws20::upcxx
{
using Clock = std::chrono::high_resolution_clock;
using DNanoseconds = std::chrono::duration<double, std::nano>;
using DMilliseconds = std::chrono::duration<double, std::milli>;

namespace detail
{
// Note: taken from namespace asc::cpp_practice_ws20::ex04
DNanoseconds measureClockResolution() {
    // We repeat the measurement multiple times and take the smallest non-zero increment of
    // consecutive time measurements as the clock resolution to avoid interference with context
    // switches.
    int repetitions = 1'000'000;

    auto time = Clock::now();

    auto duration = Clock::duration(std::numeric_limits<Clock::duration::rep>::max());

    // Measure the duration between clock ticks.
    for (int r = 0; r < repetitions; ++r) {
        auto lastTime = time;

        // Two consecutive values may yield the same result if the clock resolution is lower than
        // the latency; keep going if this happens.
        do {
            time = Clock::now();
        } while (time == lastTime);

        // Always remember the smallest non-zero time increment.
        duration = std::min(duration, time - lastTime);
    }
    return duration;
}

} // namespace detail

template <typename Func, typename ...Args>
DMilliseconds runBenchmark(Func F, Args&& ...args) {
    static_assert(std::is_invocable<Func, Args...>::value);

    // XXX: computed for different template parameters
    static DNanoseconds clockResolution = detail::measureClockResolution();
    auto start = Clock::now();

    F(std::forward<Args>(args)...);
    auto end = Clock::now();
    auto dtSeq = std::chrono::duration_cast<DMilliseconds>(end - start);

    // Note: taken from namespace asc::cpp_practice_ws20::ex04
    if (dtSeq < 100*clockResolution)
    {
        // Function call is so fast that we cannot resolve it. Run multiple iterations.
        int numRepetitions = static_cast<int>(std::min(1'000'000., dtSeq != DNanoseconds{}
                ? 100*clockResolution/dtSeq
                : 1'000'000.));

        start = Clock::now();
        for (int i = 0; i < numRepetitions; ++i) {
            F(std::forward<Args>(args)...);
        }

        end = Clock::now();
        dtSeq = std::chrono::duration_cast<DMilliseconds>(end - start) / numRepetitions;
    }
    return dtSeq;
}

// Bandwidth for sum of n array elements and in-place matrix symmetrization
template <typename T>
double bandwidthArray(DMilliseconds runtime, size_t N) {
    static_assert(std::is_arithmetic_v<T>); // float, int, double

    double gbs = sizeof(T) * (double(N) / (1024 * 1024 * 1024)); // no. gigabytes in array
    return gbs / (runtime.count() * 1000);
}

// input: ms, output: Gb/s
template <typename T>
double bandwidthStencil(DMilliseconds runtime) {
    static_assert(std::is_arithmetic_v<T>);
    // TODO
}

std::ofstream& writeCSV(std::ofstream& stream, const std::vector<std::pair<double, size_t> > &data) {
    if (!stream)
        throw std::runtime_error("could not open file for reading");
    stream << "Throughput[GB/s],Size" << std::endl;
    for (auto&& c : data) {
        stream << fmt::format("{},{}", c.first, c.second) << std::endl;
    }
    return stream;
}

} // namespace asc::pad_ws20::upcxx


#endif // BENCHMARK_H
