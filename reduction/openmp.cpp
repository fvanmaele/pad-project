#include <iostream>
#include <random>
#include <cassert>
#include <cstdio>
#include <cstddef>
#include <string>
#include <chrono>

#include <lyra/lyra.hpp>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename T>
using time_point = std::chrono::time_point<T>;
using index_t = std::ptrdiff_t;

int main(int argc, char** argv) {
    index_t N = 0; // array size
    int seed = 42; // seed for pseudo-random generator
    int iterations = 1;
    int threads = omp_get_num_threads();
    bool bench = false;
    bool write = false;
    bool show_help = false;

    auto cli = lyra::help(show_help) |
        lyra::opt(N, "size")["-N"]["--size"](
            "Size of reduced array, must be specified") |
        lyra::opt(iterations, "iterations")["--iterations"](
            "Number of iterations, default is 1") |
        lyra::opt(seed, "seed")["--seed"](
            "Seed for pseudo-random number generation, default is 42") |
        lyra::opt(threads, "threads")["-n"]["--threads"](
            "Number of threads, default is omp_get_num_threads()") |
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
    float* v = new float[N];
    double sum = 0;

    // Initialize pseudo-random number generator
    std::mt19937_64 rgen(seed);

#pragma omp parallel firstprivate(rgen)
{
    index_t block_size = N / threads;
    assert(N == block_size * threads);

    rgen.discard(block_size * omp_get_thread_num()); // advance pseudo-random number generator

#pragma omp for schedule(static)
    for (index_t i = 0; i < N; ++i) {
        v[i] = 0.5 + rgen() % 100;
    } // barrier
}
    double time = 0;
    for (int iter = 1; iter <= iterations; ++iter) {
        time_point<Clock> t = Clock::now();

    #pragma omp parallel for simd schedule(static) reduction(+:sum)
        for (index_t i = 0; i < N; ++i) { 
            sum += v[i];
        } // barrier

        Duration d = Clock::now() - t;
        time += d.count(); // time in seconds
    }
    if(bench) {
        time /= iterations;
        double throughput = N * sizeof(float) * 1e-9 / time;
        std::fprintf(stdout, "%ld,%.12f,%.12f\n", N, time, throughput);
    }    
    if (write) {
        std::cout << sum << std::endl;
    }
    delete[] v;
}
