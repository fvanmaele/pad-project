#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <numeric>
#include <vector>
#include <random>
#include <sstream>

#include "benchmark.h"

using namespace asc::pad_ws20::upcxx;

TEST_CASE("measure bandwidth of basic operations") {
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

    SECTION("vector reduction") {
        auto bench = [](const std::vector<float> &v) -> double {
            return std::accumulate<std::vector<float>::const_iterator, double>(v.begin(), v.end(), 0.0);
        };
        double count_ms = runBenchmark(bench, v).count();
        CHECK(count_ms > 0);

        std::vector<float> w(v.begin(), v.begin()+(n/3));
        double count_ms2 = runBenchmark(bench, w).count();
        CHECK(count_ms2 > 0);
        CHECK(count_ms2 != count_ms); // count_ms2 ~~ count_ms / 3

        std::vector<float> z = { v[0] };
        double count_ms3 = runBenchmark(bench, z).count();
        CHECK(count_ms3 > 0);
        CHECK(count_ms3 != count_ms);

        DMilliseconds bench_ms = runBenchmark(bench, v);
        double bw = bandwidthArray<float>(bench_ms, v.size());
        double bw_check = 4. * v.size() / 1024 / 1024 / 1024 / (1000 * bench_ms.count());
        CHECK(bw == Approx(bw_check));
        CHECK(bw > 0);
    }
}

TEST_CASE("serialization to CSV") {

}
