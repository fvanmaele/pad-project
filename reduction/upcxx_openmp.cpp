#include <iostream>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <string>
#include <chrono>
#include <algorithm>
#include <vector>
#include <limits>

#include <lyra/lyra.hpp>
#include <omp.h>
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
    int iterations = 1;
    bool show_help = false;

    auto cli = lyra::help(show_help) |
        lyra::opt(N, "size")["-N"]["--size"](
            "Size of reduced array, must be specified") |
        lyra::opt(iterations, "iterations")["--iterations"](
            "Number of iterations, default is 1") |
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

    // Sequential reduction for comparison with parallel reduction
    double sum_serial;
    {
        std::vector<float> v(N);
        std::mt19937_64 rgen(seed);
        std::generate(v.begin(), v.end(), [&rgen]() {
            return 0.5 + rgen() % 100;
        });

        // Use a reduction value with higher precision than the input values for improved
        // numerical stability. An alternative (for a `float` reduction value) is pairwise
        // or Kahan summation.
        sum_serial = std::accumulate<std::vector<float>::iterator, double>(v.begin(), v.end(), 0.0);
    }

    // BEGIN PARALLEL REGION
    upcxx::init();
    int nproc = upcxx::rank_n();
    int proc_id = upcxx::rank_me();

    // Block size for each process
    const index_t block_size = N / nproc;

    if (block_size % 2 != 0)
        throw std::invalid_argument("block size must be even");
    if (N != block_size * nproc)
        throw std::invalid_argument("array cannot be divided in same-sized blocks");

    // Allocate array, with blocks divided between processes
    float* u = new float[N];

    std::mt19937_64 rgen(seed);

#pragma omp parallel firstprivate(rgen)
{
    const int threads = omp_get_num_threads();
    const index_t block_size_omp = block_size / threads;

#pragma omp master
{
    if (block_size_omp % 2 != 0)
        throw std::invalid_argument("block size must be even (thread-local)");
    if (block_size != block_size_omp * threads)
        throw std::invalid_argument("array cannot be divided in same-sized blocks (thread-local)");
}
    rgen.discard((proc_id * threads + omp_get_thread_num()) * block_size_omp);

    // Initialize vector with pseudo-random values (consistent with serial version)
#pragma omp for schedule(static)
    for (index_t i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }
}
    // Timings for different iterations; the mean is taken later.
    std::vector<double> vt;
    vt.reserve(iterations);

    for (int iter = 1; iter <= iterations; ++iter)
    {
        // Set up a barrier before doing any timing
        upcxx::barrier();
        time_point<Clock> t = Clock::now();
        
        // Compute partial sums (threading)
        double psum(0);
#pragma omp parallel for simd schedule(static) reduction(+:psum)
        for (index_t i = 0; i < block_size; ++i) {
            psum += u[i];
        } // barrier

        // Reduce and store result on process 0
        double sum = upcxx::reduce_one(psum, upcxx::op_fast_add, 0).wait();

        if (proc_id == 0) {
            Duration d = Clock::now() - t;
            double time = d.count(); // time in seconds
            vt.push_back(time);

            // Verify against serial implementation
            if (std::abs(sum - sum_serial) > std::numeric_limits<double>::epsilon()) {
                std::cerr << "WARNING: parallel and serial sum mismatch (iteration: " << iter << ")" << std::endl
                          << sum << " vs. " << sum_serial << std::endl;
            }
        }
    }
    if (proc_id == 0) {
        double time = std::accumulate(vt.begin(), vt.end(), 0.);
        time /= vt.size(); // average time

        double throughput = N * sizeof(float) * 1e-9 / time;
        std::fprintf(stdout, "%ld,%.12f,%.12f\n", N, time, throughput);
    }
    delete[] u;

    upcxx::finalize();
    // END PARALLEL REGION
}
