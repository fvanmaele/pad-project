#include <iostream>
#include <random>
#include <cassert>
#include <cstdio>
#include <cstddef>
#include <string>
#include <chrono>
#include <vector>

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
    time_point<Clock> t{};      // time for reduction
    double psum(0);             // partial sum for current process

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
    } // barrier

    // Measure on a single thread
    if (proc_id == 0 && thread_id == 0) {
        t = Clock::now();
    }

// XXX: separate this block (support multiple iterations)
#pragma omp for simd schedule(static) reduction(+:psum)
    for (index_t i = 0; i < block_size; ++i) {
        psum += u[i];
    } // barrier
}

    // Use dist_object to communicate partial sums for final reduction
    upcxx::dist_object<double> psum_d(psum);
    upcxx::barrier();

std::cout << "(Rank " << proc_id << ")" << *psum_d << std::endl;

    if (proc_id == 0) {
		double result = *psum_d;

		// XXX; While UPCXX supports a "promise" mechanism to track completions, it is not compatible
		// to distributed objects. As a workaround, spawn and retrieve values in two seperate loops.
		// See: https://bitbucket.org/berkeleylab/upcxx/issues/452/use-of-promises-with-dist_object-rpc
		std::vector<upcxx::future<double>> futures;

		for (int k = 1; k < nproc; ++k) {
			futures.push_back(std::move(psum_d.fetch(k)));
		}
		for (int k = 1; k < nproc; ++k) {
			result += futures[k-1].wait();
		}

        if (bench) {
            // END TIMING - reduction
            Duration d = Clock::now() - t;

            double time = d.count(); // time in seconds
            double throughput = sizeof(float) * N * 1e-9 / time; // throughput in GB/s
            std::cout << N << "," << time << "," << throughput << std::endl;
        }
        if (write) {
            std::cout << result << std::endl;
        }
    }
    delete[] u;

    upcxx::finalize();
    // END PARALLEL REGION
}
