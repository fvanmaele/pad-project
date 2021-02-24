#include <iostream>
#include <random>
#include <cassert>
#include <cstdio>
#include <cstddef>
#include <string>
#include <chrono>

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

    // BEGIN PARALLEL REGION
    upcxx::init();
    int nproc = upcxx::rank_n();
    int proc_id = upcxx::rank_me();

    // Block size for each process
    const index_t block_size = N / nproc;
    assert(block_size % 2 == 0);
    assert(N == block_size * nproc);

    // Initialize array, with blocks divided between processes
    float* u = new float[N];

    std::mt19937_64 rgen(seed); // pseudo-random number generator

#pragma omp parallel firstprivate(rgen)
{
    int nthreads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();

    const index_t block_size_omp = block_size / nthreads;
    assert(block_size_omp % 2 == 0);
    assert(block_size == block_size_omp * nthreads);

    rgen.discard((proc_id * nthreads + thread_id) * block_size_omp);

#pragma omp for schedule(static)
    for (index_t i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }
}
    double time = 0;
    for (int iter = 1; iter <= iterations; ++iter)
    {
        time_point<Clock> t{};
        // Final reduction takes place on process 0, so perform timing there.
        if (proc_id == 0) {
            t = Clock::now();
        }

        // Compute partial sums (threading)
        double psum(0);
#pragma omp parallel for simd schedule(static) reduction(+:psum)
        for (index_t i = 0; i < block_size; ++i) {
            psum += u[i];
        }
        
        // Initialize distributed object for communicating partial sums.
        // Note: as there is no guarantee that every local value is constructed after the call,
        // synchronize with upcxx::barrier before proceeding.
        upcxx::dist_object<double> psum_d(psum);
        upcxx::barrier();

        // Communicate partial sums asynchronously
        if (proc_id == 0) {
            double result = *psum_d;

            // Instead of spawning and retrieving values in two separate loops, we could use the "promise"
            // mechanism to track completions.
            // See: https://bitbucket.org/berkeleylab/upcxx/issues/452/use-of-promises-with-dist_object-rpc
            std::vector<upcxx::future<double>> futures;

            for (int k = 1; k < nproc; ++k) {
                futures.push_back(std::move(psum_d.fetch(k)));
            }
            for (int k = 1; k < nproc; ++k) {
                result += futures[k-1].wait();
            }
            Duration d = Clock::now() - t;
            time += d.count(); // time in seconds

            if (write) {
                std::cout << result << std::endl;
            }
        }
        if (proc_id == 0 && bench) {
            time /= iterations;
            double throughput = N * sizeof(float) * 1e-9 / time;
            std::fprintf(stdout, "%ld,%.12f,%.12f\n", N, time, throughput);
        }
    }
    delete[] u;

    upcxx::finalize();
    // END PARALLEL REGION
}
