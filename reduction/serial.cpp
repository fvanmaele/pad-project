
#include <random>
#include <ios>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <chrono>

#include <cstdlib>
#include <getopt.h>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;
template <typename T>
using timePoint = std::chrono::time_point<T>;


int main(int argc, char** argv) {
    long N = 0;     // array size
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
                N = std::stol(optarg);
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
    
    std::vector<float> v(N);
    std::mt19937_64 rgen(seed);
    std::generate(v.begin(), v.end(), [&rgen]() {
        return 0.5 + rgen() % 100;
    });

    timePoint<Clock> t; 
    if (bench) {
        t = Clock::now();
    }
    double res = std::accumulate<std::vector<float>::iterator, double>(v.begin(), v.end(), 0.0);

    if (bench) {
        Duration d = Clock::now() - t;
        double time = d.count(); // time in seconds
        std::cout << std::fixed << time << std::endl;
    }
    if (write) {
        std::cout << res << std::endl;
    }
}
