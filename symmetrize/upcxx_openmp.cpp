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
    long dim = 0;   // amount of rows/columns
    int seed = 42;  // seed for pseudo-random generator

    struct option long_options[] = {
        { "dim",  required_argument, NULL, 'd' },
        { "seed", optional_argument, NULL, 't' },
        { NULL, 0, NULL, 0 }
    };

    int c;
    while ((c = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
        switch(c) {
            case 'd':
                dim = std::stol(optarg);
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
    if (dim <= 0)
        throw std::invalid_argument("a positive dimension is required");

    // BEGIN PARALLEL REGION
    upcxx::init();
    int nproc = upcxx::rank_n();
    int proc_id = upcxx::rank_me();

    // Block size for each process
    const long N = dim*(dim - 1) / 2;
    const long block_size = N / nproc;
    
    assert(block_size % 2 == 0);
    assert(N == block_size * nproc);

    // For symmetrization of a square matrix, we consider three arrays:
    // one holding the lower triangle, in col-major order;
    // one holding the upper triangle, in row-major order;
    // one holding the diagonal.
    // Symmetrization does not modify the diagonal, so we leave it out.
    upcxx::global_ptr<float> lower_g(upcxx::new_array<float>(block_size));
    upcxx::global_ptr<float> upper_g(upcxx::new_array<float>(block_size));
    
    assert(lower_g.is_local()); // ensure global pointer has affinity to a local process
    float* lower = lower_g.local(); // downcast to local pointer
    assert(upper_g.is_local());
    float* upper = upper_g.local();
    
    // Initialize upper and lower triangle
    std::mt19937_64 rgen(seed);
    for (long i = 0; i < block_size; ++i) {
        lower[i] = 0.5 + rgen() % 100;
        upper[i] = 1.0 + rgen() % 100;
    }

    // Symmetrize matrix (SAXPY over lower and upper triangle).
    // We only require a single for loop because lower and upper triangle
    // are stored symmetrically (col-major and row-major, respectively)
    #pragma omp parallel for simd shared(lower, upper)
    for (long i = 0; i < block_size; ++i) {
        float s = (lower[i] + upper[i]) / 2.;
        lower[i] = s;
        upper[i] = s;
    }

    // TODO: serialize transposed matrix (process-by-process, cf. reduction)

    upcxx::delete_array(lower_g);
    upcxx::delete_array(upper_g);

    upcxx::finalize();
    // END PARALLEL REGION
}