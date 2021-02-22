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
    int nproc = omp_get_num_threads();
    int proc_id = omp_get_thread_num();
    
    index_t block_size = N / nproc;
    assert(N == block_size * nproc);

    rgen.discard(block_size * proc_id); // advance pseudo-random number generator

#pragma omp for schedule(static)
    for (index_t i = 0; i < N; ++i)
    {
        v[i] = 0.5 + rgen() % 100;
    } // barrier

// XXX: separate out next blocks (support multiple iterations)
    time_point<Clock> t;
    if (bench && omp_get_thread_num() == 0) // measure time on a single thread
    {
        t = Clock::now();
    }

#pragma omp for simd schedule(static) reduction(+:sum)
    for (index_t i = 0; i < N; ++i)
    { 
        sum += v[i];
    } // barrier

    if (bench && omp_get_thread_num() == 0) {
        Duration d = Clock::now() - t;
        double time = d.count(); // time in seconds
        std::cout << std::fixed << time << std::endl;
    }
}

    if (write) {
        std::cout << std::defaultfloat << sum << std::endl;
    }
    delete[] v;
}
