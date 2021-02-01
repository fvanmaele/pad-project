
#include <random>
#include <iomanip>
#include <iostream>
#include <string>
#include <chrono>

#include <cassert>
#include <cstdlib>
#include <getopt.h>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;
template <typename T>
using timePoint = std::chrono::time_point<T>;


int main(int argc, char** argv) {
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
    float* v = new float[N];
    double sum = 0;

    // Initialize pseudo-random number generator
    std::mt19937_64 rgen(seed);

#pragma omp parallel firstprivate(rgen)
{
    int nproc = omp_get_num_threads();
    int proc_id = omp_get_thread_num();
    
    int64_t block_size = N / nproc;
    assert(N == block_size * nproc);
    rgen.discard(block_size * proc_id); // advance pseudo-random number generator

#pragma omp for schedule(static)
    for (int64_t i = 0; i < N; ++i)
    {
        v[i] = 0.5 + rgen() % 100;
    } // barrier

    timePoint<Clock> t;
    if (bench && omp_get_thread_num() == 0) // measure time on a single thread
    {
        t = Clock::now();
    }

#pragma omp for simd schedule(static) reduction(+:sum)
    for (int64_t i = 0; i < N; ++i)
    { 
        sum += v[i];
    } // barrier

    if (bench && omp_get_thread_num() == 0) {
        Duration d = Clock::now() - t;
        double time = d.count(); // time in seconds
        std::cout << std::fixed << std::setprecision(16) << time << std::endl;
    }
}
// END PARALLEL REGION
    if (write) {
        std::cout << std::defaultfloat << std::setprecision(12) << sum << std::endl;
    }
    delete[] v;
}
