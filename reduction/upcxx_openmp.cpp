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
    const index_t block_size_omp = block_size / threads;
    assert(block_size_omp % 2 == 0);
    assert(block_size == block_size_omp * threads);

    rgen.discard((proc_id * threads + omp_get_thread_num()) * block_size_omp);

#pragma omp for schedule(static)
    for (index_t i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }
}
    // According to 14.3 (Specification), the name carried by the distributed object (here psum_d)
    // may not exist yet in all processes after the call. To avoid this, we use the described
    // "asynchronous point-to-point" approach, implicitly used when dist_object<T>& arguments
    // are given to an RPC (in particular, dist_object::fetch).
    upcxx::dist_object<double> psum_d(0);

    // Timings for different iterations; the mean is taken later.
    std::vector<double> vt;
    vt.reserve(iterations);
    for (int iter = 1; iter <= iterations; ++iter)
    {
        // To reduce latency, we spawn and retrieve values in two separate loops. An alternative
        // is to use the upcxx "promise" mechanism for tracking completions.
        // See: https://bitbucket.org/berkeleylab/upcxx/issues/452/use-of-promises-with-dist_object-rpc
        std::vector<upcxx::future<double>> futures;
        futures.reserve(nproc);
        
        // Compute partial sums (threading)
        time_point<Clock> t = Clock::now();
        double psum(0);

#pragma omp parallel for simd schedule(static) reduction(+:psum)
        for (index_t i = 0; i < block_size; ++i) {
            psum += u[i];
        }
        *psum_d = psum;

        // Communicate partial sums asynchronously
        if (proc_id == 0) {
            double result = *psum_d;

            for (int k = 1; k < nproc; ++k) {
                futures.push_back(std::move(psum_d.fetch(k)));
            }
            for (int k = 1; k < nproc; ++k) {
                result += futures[k-1].wait();
            }
            Duration d = Clock::now() - t;
            double time = d.count(); // time in seconds
            vt.push_back(time);

            if (write) {
                std::cout << result << std::endl;
            }
        }
        if (proc_id == 0 && bench) {
            for (auto&& time: vt) {
                double throughput = N * sizeof(float) * 1e-9 / time;
                std::fprintf(stdout, "%ld,%.12f,%.12f\n", N, time, throughput);
            }
        }
        upcxx::barrier();
    }
    delete[] u;

    upcxx::finalize();
    // END PARALLEL REGION
}
