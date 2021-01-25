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
        throw std::invalid_argument("a positive array size is required");
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
    for (long i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }

    // Create a reduction value for each process (universal name, local value).
    // Each local value can be accessed with operator* or operator->, but there
    // is no guarantee that every local value is constructed after the call.
    upcxx::dist_object<double> psum_d(0);
    upcxx::barrier();

    for (long i = 0; i < block_size; ++i) {
        *psum_d += u[i];
    }

    // Ensure all partial sums are available
    upcxx::barrier();
    std::cout << *psum_d << " (Rank " << proc_id << ")" << std::endl;

    // Reduce partial sums through dist_object::fetch (communication) on master process
    if (proc_id == 0) {
        // partial sum for process 0
        double res(*psum_d);

        // partial sums for remaining processes
        for (int k = 1; k < upcxx::rank_n(); ++k) {
            double psum = psum_d.fetch(k).wait();
            res += psum;
        }
        std::cout << res << std::endl;
    }

    upcxx::delete_array(u_g);

    upcxx::finalize();
    // END PARALLEL REGION
}