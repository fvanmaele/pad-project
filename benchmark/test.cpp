#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <numeric>
#include <vector>
#include <random>

#include "benchmark.h"

using namespace asc::pad_ws20::upcxx;

TEST_CASE("timing of basic operations") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist{};

    // Initialize random elements
    constexpr ptrdiff_t n = 1000000;
    std::vector<float> v{};
    v.reserve(n);

    for (ptrdiff_t i = 0; i < n; ++i) {
        v.push_back(dist(gen));
    }

    // Benchmark operation
    auto bench = [&v]() -> float {
        return std::accumulate<std::vector<float>::iterator, double>(v.begin(), v.end(), 0.0);
    };
    double count_ms = runBenchmark(bench);
    CHECK(count_ms > 0);

    v.clear();
    for (ptrdiff_t i = 0; i < n/3; ++i) {
        v.push_back(dist(gen));
    }
    double count_ms2 = runBenchmark(bench);
    CHECK(count_ms2 > 0);
    CHECK(count_ms2 != count_ms); // count_ms2 ~~ count_ms / 3

    v.clear();
    v.push_back(dist(gen));
    double count_ms3 = runBenchmark(bench);
    CHECK(count_ms3 > 0);
    CHECK(count_ms3 != count_ms);
}
