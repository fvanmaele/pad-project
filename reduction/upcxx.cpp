#include <ios>
#include <iostream>
#include <random>
#include <cassert>
#include <utility>
#include <string>
#include <chrono>

#include <cstdlib>
#include <getopt.h>
#include <upcxx/upcxx.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;
template <typename T>
using timePoint = std::chrono::time_point<T>;


int main(int argc, char** argv) 
{
    int64_t N = 0;     // array size
    int seed = 42;  // seed for pseudo-random generator
    bool bench = false;
    bool write = false;

    struct option long_options[] = {
        { "size", required_argument, NULL, 's' },
        { "seed", required_argument, NULL, 't' },
        { "bench", no_argument, NULL, 'b' },
        { "write", no_argument, NULL, 'w' },
        { NULL, 0, NULL, 0 }
    };

    int c;
    while ((c = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
        switch(c) {
            case 's':
                N = std::stoll(optarg);
                break;
            case 't':
                seed = std::stoi(optarg);
                break;
            case 'b':
                bench = true;
                break;
            case 'w':
                write = true;
                break;
            case '?':
                break;
            default:
                std::terminate();
        }
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
    const int64_t block_size = N / nproc;
    assert(block_size % 2 == 0);
    assert(N == block_size * nproc);

    // Initialize array, with blocks divided between processes
    upcxx::global_ptr<float> u_g(upcxx::new_array<float>(block_size));
    assert(u_g.is_local()); // ensure global pointer has affinity to a local process
    float* u = u_g.local(); // downcast to local pointer

    // Fill with random values (using discard to ensure pseudo-random values
    // across the full array)
    std::mt19937_64 rgen(seed);
    rgen.discard(proc_id * block_size);
    for (int64_t i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }

    // BEGIN TIMING - reduction
    timePoint<Clock> t{};
    if (proc_id == 0 && bench) {
        // Timing is done on the master process only (as this is where the final reduction
        // of partial sums will take place)
        t = Clock::now();
    }
    
    // Create a reduction value for each process (universal name, local value).
    // Each local value can be accessed with operator* or operator->, but there
    // is no guarantee that every local value is constructed after the call.
    upcxx::dist_object<double> psum_d(0);
    upcxx::barrier();

    // Compute partial sums and ensure they are available
    for (int64_t i = 0; i < block_size; ++i) {
        *psum_d += u[i];
    }
    upcxx::barrier();

    if (write) {
        std::cout << *psum_d << " (Rank " << proc_id << ")" << std::endl;
    }

    // Reduce partial sums through dist_object::fetch (communication) on master process.
    // Alternative (with reduction in random order): upcxx::reduce_all() or reduce_one()
    if (proc_id == 0) {
        // partial sum for process 0
        double res(*psum_d);

        // partial sums for remaining processes (in ascending order)
        for (int k = 1; k < upcxx::rank_n(); ++k) {
            double psum = psum_d.fetch(k).wait();
            res += psum;
        }
        if (bench) {
            // END TIMING - reduction
            Duration d = Clock::now() - t;
            double time = d.count(); // time in seconds
            std::cout << std::fixed << time << std::endl;
        }
        if (write) {
            std::cout << res << std::endl;
        }
    }
    upcxx::delete_array(u_g);

    upcxx::finalize();
    // END PARALLEL REGION
}
