#include <iostream>
#include <random>
#include <cstdio>
#include <cstddef>
#include <string>
#include <algorithm>
#include <vector>
#include <chrono>

#include <lyra/lyra.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename T>
using time_point = std::chrono::time_point<T>;
using index_t = std::ptrdiff_t;

int main(int argc, char** argv) {
    index_t N = 0;     // array size
    int seed = 42;  // seed for pseudo-random generator
    bool bench = false;
    bool write = false;
    bool show_help = false;

    auto cli = lyra::help(show_help) |
        lyra::opt(N, "size")["-N"]["--size"](
            "Size of reduced array, must be specified") |
        lyra::opt(seed, "seed")["--seed"](
            "Seed for pseudo-random number generation, default is 42") |
        lyra::opt(bench)["--bench"](
            "Enable benchmarking") |
        lyra::opt(write)["--write"](
            "Print reduction value to standard output");
    auto result = cli.parse({argc, argv});
    
    if (!result) {
		std::cerr << "Error in command line: " << result.errorMessage()
			  << std::endl;
		exit(1);
	}
	if (show_help) {
		std::cout << cli << std::endl;
		exit(0);
	}

    if (N <= 0) {
        std::cerr << "a positive array size is required (specify with --size)" << std::endl;
        std::exit(1);
    }
    
    std::vector<float> v(N);
    std::mt19937_64 rgen(seed);
    std::generate(v.begin(), v.end(), [&rgen]() {
        return 0.5 + rgen() % 100;
    });

    time_point<Clock> t; 
    if (bench) {
        t = Clock::now();
    }
    double res = std::accumulate<std::vector<float>::iterator, double>(v.begin(), v.end(), 0.0);

    if (bench) {
        Duration d = Clock::now() - t;
        double time = d.count(); // time in seconds
        double throughput = N * sizeof(float) * 1e-9 / time;
        std::fprintf(stdout, "%ld,%.12f,%.12f\n", N, time, throughput);
    }
    if (write) {
        std::cout << res << std::endl;
    }
}
