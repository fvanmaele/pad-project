#include <vector>
#include <random>
#include <iostream>
#include <string>

#include <cstdlib>
#include <getopt.h>

int main(int argc, char** argv) {
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
    if (dim <= 0) {
        std::cerr << "a positive array size is required (specify with --size)" << std::endl;
        std::exit(1);
    }

    const long N = dim*(dim - 1) / 2;
    std::vector<float> lower(dim);
    std::vector<float> upper(dim);

    std::mt19937_64 rgen(seed);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) {
        lower[i] = 0.5 + rgen() % 100;
        upper[i] = 1.0 + rgen() % 100;
    }

#pragma omp parallel for shared(lower, upper) schedule(static)
    for (long i = 0; i < N; ++i) {
        float s = (lower[i] + upper[i]) / 2.;
        lower[i] = s;
        upper[i] = s;
    }

    // TODO: serialize transposed matrix
}