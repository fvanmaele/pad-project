#include <iostream>
#include <random>
#include <cassert>
#include <utility>
#include <string>

#include <cstdlib>
#include <getopt.h>
#include <upcxx/upcxx.hpp>

int main(int argc, char** argv) 
{
    long N = 0;     // array size
    int seed = 42;  // seed for pseudo-random generator

    struct option long_options[] = {
        { "size", required_argument, NULL, 's' },
        { "seed", optional_argument, NULL, 't' },
        { NULL, 0, NULL, 0 }
    };

    int c;
    while ((c = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
        switch(c) {
            case 's':
                N = std::stol(optarg);
                break;
            case 't':
                seed = std::stoi(optarg);
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
    const long block_size = N / nproc;
    assert(block_size % 2 == 0);
    assert(N == block_size * nproc);

    // Initialize array, with blocks divided between processes
    upcxx::global_ptr<float> u_g(upcxx::new_array<float>(block_size));
    assert(u_g.is_local()); // ensure global pointer has affinity to a local process
    float* u = u_g.local(); // downcast to local pointer
    
    std::mt19937_64 rgen(seed);
    // XXX: discard according to rank
    // rgen.discard(upcxx::rank_me() * block)
//#pragma omp parallel for schedule(static)
    for (long i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }

    // Create a partial value for each process, later reduced with upcxx::reduce_all.
    double psum(0);
#pragma omp parallel for reduction(+: psum)
    for (long i = 0; i < block_size; ++i) {
        psum += u[i];
    }

    // Ensure all partial sums are available.
    upcxx::barrier();
    std::cout << psum << " (Rank " << proc_id << ")" << std::endl;

    // Reduce partial sums
    double res = upcxx::reduce_all(psum, upcxx::op_fast_add).wait();
    if (proc_id == 0) {
        std::cout << res << std::endl;
    }
    
    upcxx::delete_array(u_g);

    upcxx::finalize();
    // END PARALLEL REGION
}