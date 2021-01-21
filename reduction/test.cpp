#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <vector>
#include <random>
#include <numeric>

#include "accumulate.h"

using namespace asc::pad_ws20::upcxx;

TEST_CASE("different result and summand type") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.5);

    constexpr ptrdiff_t n = 10000000;
    std::vector<float> v{};
    v.reserve(n);
    for (ptrdiff_t i = 0; i < n; ++i) {
         // simple random walk
        dist(gen) ? v.push_back(1) : v.push_back(-1);
    }

    SECTION("naive summation") {
        double result = sum<float, double>(v.data(), v.size(), 0.);
        double result_std = std::accumulate<std::vector<float>::iterator, double>(v.begin(), v.end(), 0.0);
        CHECK(result == Approx(result_std));
    }

    SECTION("pairwise summation") {
        double result_pw = sum_pairwise<float, double, 1000>(v.data(), v.size(), 0.);
        double result = sum<float, double>(v.data(), v.size(), 0.);
        CHECK(result_pw == Approx(result));
    }

    SECTION("Kahan summation") {
        double result_kahan = sum_kahan<float>(v.data(), v.size(), 0.);
        double result = sum<float, double>(v.data(), v.size(), 0.);
        CHECK(result_kahan == Approx(result));
    }
}
