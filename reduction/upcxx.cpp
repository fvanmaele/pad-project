#include <iostream>
#include <random>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <string>
#include <chrono>
#include <vector>

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
    std::vector<float> u(block_size);

    // Fill with random values (consistent with sequential version)
    std::mt19937_64 rgen(seed);
    rgen.discard(proc_id * block_size);
    for (index_t i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }

    double time = 0;
    for (int iter = 1; iter <= iterations; ++iter) {
        time_point<Clock> t{};
        // Timing is done on process 0 only (as this is where the final reduction
        // of partial sums will take place)
        if (proc_id == 0) {
            t = Clock::now();
        }

        // Compute partial sums and assign to local value of distributed object
        double psum(0);
        for (index_t i = 0; i < block_size; ++i) {
            psum += u[i];
        }
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
    }
    if (proc_id == 0 && bench) {
        time /= iterations;
        double throughput = N * sizeof(float) * 1e-9 / time;
        std::fprintf(stdout, "%ld,%.12f,%.12f\n", N, time, throughput);
    }

    upcxx::finalize();
    // END PARALLEL REGION
}
