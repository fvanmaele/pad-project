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
using time_point = std::chrono::time_point<T>;
using index_t = std::ptrdiff_t;

int main(int argc, char** argv) 
{
    int seed = 42;  // seed for pseudo-random generator
    bool bench = false;
    bool write = false;
    const char* file_path = "upcxx_stencil.txt";

    index_t size_x = 0;
    index_t size_y = 0;
    index_t size_z = 0;
    int radius = 2;
    int steps = 1;

    struct option long_options[] = {
        // dimensions
        { "size_x", required_argument, NULL, 'x' },
        { "size_y", required_argument, NULL, 'y' },
        { "size_z", required_argument, NULL, 'z' },
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
            case 'x': size_x = std::stoll(optarg); break;
            case 'y': size_y = std::stoll(optarg); break;
            case 'z': size_z = std::stoll(optarg); break;
            case 'r': radius = std::stoll(optarg); break;
            case 's': steps  = std::stoll(optarg); break;
            case 'S': seed   = std::stoll(optarg); break;
            case 'b': bench  = true; break;
            case 'w': 
                write = true;
                if (optarg) file_path = optarg;
                break;
            default: 
                std::terminate();
        }
    }
    if (size_x <= 0) {
        std::cerr << "the x-dimension must be positive (specify with --size_x)" << std::endl;
        std::exit(1);
    }
    if (size_y <= 0) {
        std::cerr << "the x-dimension must be positive (specify with --size_x)" << std::endl;
        std::exit(1);
    }
    if (size_z <= 0) {
        std::cerr << "the x-dimension must be positive (specify with --size_x)" << std::endl;
        std::exit(1);
    }
    if (radius <= 0) {
        std::cerr << "the radius must be positive" << std::endl;
        std::exit(1);
    }
    if (steps <= 0) {
        std::cerr << "the amount of steps must be positive" << std::endl;
        std::exit(1);
    }
    // rough overflow checks (XXX: does not include radius)
    assert(size_y < std::numeric_limits<index_t>::max() / size_z / size_x);
    assert(size_x < std::numeric_limits<index_t>::max() / size_y / size_z);
    assert(size_z < std::numeric_limits<index_t>::max() / size_x / size_y);

    // BEGIN PARALLEL REGION
    upcxx::init();
    const upcxx::intrank_t proc_n = upcxx::rank_n();
    const upcxx::intrank_t proc_id = upcxx::rank_me();

    // Get the bounds for the local panel, assuming the number of processes divides the
    // element size into an even block size.
    const index_t N = size_x * size_y * size_z;
    const index_t block = N / proc_n;
    assert(block % 2 == 0);
    assert(N == block * proc_n);

    // Allocate space for 3-dimensional input arrays, including zero padding for ghost cells.
    // Ghost cells are accessed in a periodic fashion: process 0 has process n-1 and 1 
    // as left and right neighbors, respectively. The array is split in the z-axis, such that 
    // the amount of ghost cells is determined by 2* radius * size_x * size_y: on the upper
    // and lower border of each block, an element needs to access 2 * radius cells (z direction).
    // XXX: use upcxx::local_team() to reducde overhead when accessing elements on the same node
    assert(radius % 2 == 0);
    const index_t n_ghost_offset = radius*size_x*size_y;    // bound for ghost cells, z radius
    const index_t n_local = block + 2*n_ghost_offset;       // elements + ghost cells

    // FDTD: E (electric field), used as input array on even steps and output array on uneven steps.
    // The alternation between input and output array allows to implement the stencil as a gather.
    upcxx::dist_object<upcxx::global_ptr<float>> E_g = upcxx::new_array<float>(n_local);
    assert(E_g->is_local());
    float* E = E_g->local();

    // Initialize E with pseudo-random numbers for first step (t = 0), independent of job size
    std::mt19937_64 rgen(seed);
    rgen.discard(proc_id * block);
    for (index_t i = n_ghost_offset; i < n_local - n_ghost_offset; ++i) {
        E[i] = 0.5 + rgen() % 100;
    }

    // FDTD: H (magnetic field), used as input array on uneven steps and output array on even steps.
    upcxx::dist_object<upcxx::global_ptr<float>> H_g = upcxx::new_array<float>(n_local);
    assert(H_g->is_local());
    float* H = H_g->local();

    // Auxiliary arrays for FDTD (coefficients)
    // XXX: these can be shared on the same node between processes, i.e. copies are only required
    // for different nodes (see upcxx::local_team(), upcxx::broadcast)
    upcxx::global_ptr<float> coeff_g = upcxx::new_array<float>(radius+1);
    assert(coeff_g.is_local());
    float* coeff = coeff_g.local();
    
    for (index_t i = 0, k = radius+1; i < k; ++i) {
        coeff[i] = 0.2 * i + 0.5;
    }

    // NOTE: while ghost cells are not required for the coefficients, we use the same size
    // of E and H to avoid indexing errors.
    upcxx::global_ptr<float> perm_g = upcxx::new_array<float>(n_local);
    assert(perm_g.is_local());
    float* perm = perm_g.local();
    
    for (index_t i = n_ghost_offset; i < n_local - n_ghost_offset; ++i) {
        perm[i] = 1; // XXX: use random values (with correct per-process offset)
    }

    // Wait for initialization to complete
    upcxx::barrier();

    // 3-dimensional arrays are accessed through offsets on a regular 1-dimensional array.
    index_t size_xy = size_x * size_y;
    auto ind3 = [size_x, size_xy](index_t x, index_t y, index_t z) { 
        return (z * size_xy) + (y * size_x) + x;
    };

    // Fetch the left and right pointers for the ghost cells.
    // NOTE: We define neighbors to be periodic, i.e. process 0 has process n-1 as left neighbor,
    // and process n-1 has process 0 as right neighbor.
    upcxx::intrank_t l_nbr = (proc_id + proc_n - 1) % proc_n;
    upcxx::intrank_t r_nbr = (proc_id + 1) % proc_n;

    // Begin FDTD
    time_point<Clock> t{};
    if (proc_id == 0 && bench) { // benchmark on the first process
        t = Clock::now();
    }
    for (int t = 0; t < steps; ++t) {
        // XXX: use the references below to avoid duplicating code
        //upcxx::dist_object<upcxx::global_ptr<float>>& input_g  = (t&1 == 0) ? E_g : H_g;
        //upcxx::dist_object<upcxx::global_ptr<float>>& output_g = (t&1 == 0) ? H_g : E_g;

        if ((t & 1) == 0) {  // t even: E as input, H as output
            upcxx::global_ptr<float> E_L;
            upcxx::global_ptr<float> E_R;
            // XXX: Because the fetch function is asynchronous, we have to synchronize on completion,
            // using a call to wait(). Later, we will see how to overlap asynchronous operations, that
            // is, when communication is split-phased.
            if (proc_id == 0) {
                // Lower ghost cells are zero (lower boundary of domain), fetch of left nbr not required
                E_R = E_g.fetch(r_nbr).wait();
                upcxx::rget(E_R + n_ghost_offset, E_g + n_ghost_offset + block, n_ghost_offset).wait();
            } 
            else if (proc_id == proc_n - 1) {
                // Upper ghost cells are zero (upper boundary of domain), fetch of right nbr not required
                E_L = E_g.fetch(l_nbr).wait();
                upcxx::rget(E_L + block, E_g, n_ghost_offset).wait();
            } 
            else {
                // Retrieve both lower and upper ghost cells from neighbors
                E_L = E_g.fetch(l_nbr).wait();
                upcxx::rget(E_L + block, E_g, n_ghost_offset).wait();
                E_R = E_g.fetch(r_nbr).wait();
                upcxx::rget(E_R + n_ghost_offset, E_g + n_ghost_offset + block, n_ghost_offset).wait();
            }
            // TODO: test if boundary conditions are correct (e.g. dump arrays)
            // rget(global_ptr <T> src , T *dest , std:: size_t count)

            // Compute values

        } else {  // t odd: H as input, E as input
            // rget(global_ptr <T> src , T *dest , std:: size_t count)
            // TODO: test if boundary conditions are correct (e.g. dump arrays
            if (proc_id == 0) {
                upcxx::global_ptr<float> H_R = H_g.fetch(r_nbr).wait();
                upcxx::rget(H_R + n_ghost_offset, H_g + n_ghost_offset + block, n_ghost_offset).wait();
            } 
            else if (proc_id == proc_n - 1) {
                upcxx::global_ptr<float> H_L = H_g.fetch(l_nbr).wait();
                upcxx::rget(H_L + block, H_g, n_ghost_offset).wait();
            } 
            else {
                upcxx::global_ptr<float> H_L = H_g.fetch(l_nbr).wait();
                upcxx::rget(H_L + block, H_g, n_ghost_offset).wait();
                upcxx::global_ptr<float> H_R = H_g.fetch(r_nbr).wait();
                upcxx::rget(H_R + n_ghost_offset, H_g + n_ghost_offset + block, n_ghost_offset).wait();
            }
            
            // Compute values
        }
    }

    upcxx::finalize();
    // END PARALLEL REGION
}