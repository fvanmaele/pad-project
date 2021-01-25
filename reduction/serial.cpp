
#include <random>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

#include <cstdlib>
#include <getopt.h>

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
    
    std::vector<float> v(N);
    std::mt19937_64 rgen(seed);
    std::generate(v.begin(), v.end(), [&rgen]() {
        return 0.5 + rgen() % 100;
    });

    double res = std::accumulate<std::vector<float>::iterator, double>(v.begin(), v.end(), 0.0);
    std::cout << res << std::endl;
}