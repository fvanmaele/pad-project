#include <iostream>
#include <random>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <limits>

#include <lyra/lyra.hpp>
#include <upcxx/upcxx.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename T>
using time_point = std::chrono::time_point<T>;
using index_t = std::ptrdiff_t;

int main(int argc, char** argv) 
{
    index_t N = 0; // array size
    int seed = 42; // seed for pseudo-random generator
    int iterations = 1; // repeats when using benchmark
    bool write = false;
    bool bench = false;
    bool show_help = false;

    auto cli = lyra::help(show_help) |
        lyra::opt(N, "size")["-N"]["--size"](
            "Size of reduced array, must be specified") |
        lyra::opt(iterations, "iterations")["--iterations"](
            "Number of iterations, default is 1") |
        lyra::opt(write)["--write"](
            "Print reduction value to standard output") |
        lyra::opt(bench)["--bench"](
            "Print benchmarks to standard output") |
        lyra::opt(seed, "seed")["--seed"](
            "Seed for pseudo-random number generation, default is 42");
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

    // BEGIN PARALLEL REGION
    upcxx::init();
    int nproc = upcxx::rank_n();
    int proc_id = upcxx::rank_me();

    // Block size for each process
    const index_t block_size = N / nproc;
    assert(block_size % 2 == 0);
    assert(N == block_size * nproc);

    // Initialize array, with blocks divided between processes
    std::vector<float> u(block_size);

    // Fill with random values (consistent with sequential version)
    std::mt19937_64 rgen(seed);
    rgen.discard(proc_id * block_size);
    for (index_t i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }

    // Timings for different iterations, of which the mean is taken.
    std::vector<double> vt;
    vt.reserve(iterations);
    // Reduction
    for (int iter = 1; iter <= iterations; ++iter) 
    {
        // Set a barrier before doing any timing
        upcxx::barrier();
        time_point<Clock> t = Clock::now();
        
        // Compute partial sums and reduce on process 0
        double psum(0);
        for (index_t i = 0; i < block_size; ++i) {
            psum += u[i];
        }
        double sum = upcxx::reduce_one(psum, upcxx::op_fast_add, 0).wait();

        if (proc_id == 0) {
            Duration d = Clock::now() - t;
            double time = d.count(); // time in seconds
            vt.push_back(time);
        }

        if (write) {
            std::cout << sum << std::endl;
        }
    }
    if (proc_id == 0) {
        double time = std::accumulate(vt.begin(), vt.end(), 0.);
        time /= vt.size(); // average time

        if (bench) {
            double throughput = N * sizeof(float) * 1e-9 / time;
            std::fprintf(stdout, "%ld,%.12f,%.12f\n", N, time, throughput);
        }
    }
    upcxx::finalize();
    // END PARALLEL REGION
}
