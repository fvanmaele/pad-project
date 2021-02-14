#include <random>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>

#include <getopt.h>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename T>
using time_point = std::chrono::time_point<T>;
using index_t = std::ptrdiff_t;

void requires_positive(index_t x, const char* msg) {
    if (x <= 0) {
        std::fprintf(stdout, "%s", msg);
        std::exit(1);
    }
}

int main(int argc, char** argv) 
{
    int seed = 42;  // seed for pseudo-random generator
    bool bench = false;
    bool write = false;
    const char* file_path = "openmp_stencil.txt";

    index_t dim_x = 0;
    index_t dim_y = 0;
    index_t dim_z = 0;
    int radius = 2;
    int steps = 1;

    // TODO: use Lyra
    struct option long_options[] = {
        // dimensions
        { "dim_x", required_argument, NULL, 'x' },
        { "dim_y", required_argument, NULL, 'y' },
        { "dim_z", required_argument, NULL, 'z' },
        { "radius", required_argument, NULL, 'r' },
        { "steps", required_argument, NULL, 's' },
        // program options
        { "seed", required_argument, NULL, 'S' },
        { "bench", no_argument, NULL, 'b' },
        { "write", optional_argument, NULL, 'w' },
        { NULL, 0, NULL, 0 }
    };

    int c;
    while ((c = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
        switch(c) {
        case 'x':
            dim_x = std::stoll(optarg);
            break;
        case 'y':
            dim_y = std::stoll(optarg);
            break;
        case 'z':
            dim_z = std::stoll(optarg);
            break;
        case 'r':
            radius = std::stoi(optarg);
            break;
        case 's':
            steps = std::stoi(optarg);
            break;
        case 'S':
            seed = std::stoi(optarg);
            break;
        case 'b':
            bench = true;
            break;
        case 'w':
            write = true;
            if (optarg)
                file_path = optarg;
            break;
        default:
            std::terminate();
        }
    }
    requires_positive(dim_x, "the x-dimension must be positive (specify with --dim_x)");
    requires_positive(dim_y, "the y-dimension must be positive (specify with --dim_y)");
    requires_positive(dim_z, "the z-dimension must be positive (specify with --dim_z)");
    requires_positive(radius, "the radius must be positive");
    requires_positive(steps, "the amount of steps must be positive");

    std::mt19937_64 rgen(seed);

// BEGIN PARALLEL REGION
#pragma omp parallel firstprivate(rgen)
{
    const int proc_n = omp_get_num_threads();
    const int proc_id = omp_get_thread_num();

    // Get the bounds for the local panel, assuming the number of processes divides the
    // element size into an even block size.
    const index_t N = dim_x * dim_y * dim_z;
    const index_t n_block = N / proc_n;
    assert(n_block % 2 == 0);
    assert(N == n_block * proc_n);

    // TODO

}
// END PARALLEL REGION
}