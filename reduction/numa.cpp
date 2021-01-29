
#include <random>
#include <iostream>
#include <string>

#include <cassert>
#include <cstdlib>
#include <getopt.h>
#include <omp.h>

int main(int argc, char** argv) {
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
    float* v = new float[N];
    double sum = 0;

    // Initialize pseudo-random number generator
    std::mt19937_64 rgen(seed);

#pragma omp parallel firstprivate(rgen)
{
    int nproc = omp_get_num_threads();
    int proc_id = omp_get_thread_num();
    
    long block_size = N / nproc;
    assert(N == block_size * nproc);
    rgen.discard(block_size * proc_id); // advance pseudo-random number generator

#pragma omp for schedule(static)
    for (long i = 0; i < N; ++i)
    {
        v[i] = 0.5 + rgen() % 100;
    } // barrier

#pragma omp for simd schedule(static) reduction(+:sum)
    for (long i = 0; i < N; ++i)
    { 
        sum += v[i];
    } // barrier
}
// END PARALLEL REGION

    std::cout << sum << std::endl;
    delete[] v;
}