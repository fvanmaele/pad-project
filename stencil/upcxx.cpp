#include <iostream>
#include <random>
#include <cassert>
#include <cstdlib>
#include <string>
#include <chrono>

#include <lyra/lyra.hpp>

#include "include/upcxx.hpp"
#include "include/stencil.hpp"
#include "include/stencil-upcxx.hpp"
#include "include/stencil-print.hpp"

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename T>
using time_point = std::chrono::time_point<T>;

template <typename ...Ns>
bool is_positive(Ns... args) {
    return ((args > 0) && ...);
}

int main(int argc, char** argv) 
{
    int seed = 42;  // seed for pseudo-random generator
    bool bench = false;
    bool write = false;
    bool show_help = false;
    const char* file_path = "upcxx_stencil.txt";
    const char* file_path_steps = "upcxx_stencil_steps.txt";
    const char* file_path_steps_cell = "upcxx_stencil_steps_cell.txt";

    index_t dim_x = 32;
    index_t dim_y = 32;
    index_t dim_z = 32;
    int radius = 4;
    int steps = 5;

    auto cli = lyra::help(show_help) |
        lyra::opt(dim_x, "dim_x")["-x"]["--dim_x"](
            "Size of domain (x-dimension), default is 32") |
        lyra::opt(dim_y, "dim_y")["-y"]["--dim_y"](
            "Size of domain (y-dimension), default is 32") |
        lyra::opt(dim_z, "dim_z")["-z"]["--dim_z"](
            "Size of domain (z-dimension), default is 32") |
        lyra::opt(radius, "radius")["-r"]["--radius"](
            "Stencil radius, default is 4") |
        lyra::opt(steps, "steps")["-t"]["--steps"](
            "Number of time steps, default is 5") |
        lyra::opt(bench)["--bench"](
            "Enable benchmarking") |
        lyra::opt(seed, "seed")["--seed"](
            "Seed for pseudo-random number generation, default is 42") |
        lyra::opt(write)["--write"](
            "Write out array contents to file");
    auto result = cli.parse({argc, argv});
    
    if (!is_positive(dim_x, dim_y, dim_z, radius, steps)) {
        std::cerr << "Arguments must be positive" << std::endl;
        exit(1);
    }
    if (!result) {
		std::cerr << "Error in command line: " << result.errorMessage()
			  << std::endl;
		exit(1);
	}
	if (show_help) {
		std::cout << cli << std::endl;
		exit(0);
	}

    // BEGIN PARALLEL REGION
    upcxx::init();
    const upcxx::intrank_t proc_n = upcxx::rank_n();
    const upcxx::intrank_t proc_id = upcxx::rank_me();

    // We partition the stencil arrays in the z-axis. Splits in the x- and y-axis are avoided 
    // to reduce communication costs between nodes (i.e. x/y tiling should be done locally, 
    // through broadcasting or a threading model).
    const index_t dim_zi = dim_z / proc_n;
    assert(dim_z == dim_zi * proc_n);
    const index_t n_block = dim_x * dim_y * dim_zi;

    // Checks that the size of the ghost cells does not exceed the size of the process block
    // (e.g. {4,4,4} with 4 processes (or {4,4,1} per process) and radius 2)
    assert(dim_zi >= radius);

    // Add zero padding for ghost cells (communication) and neighbor access on the domain border.
    const index_t Nx = dim_x + 2*radius;
    const index_t Ny = dim_y + 2*radius;
    const index_t Nz = dim_zi + 2*radius;
    const index_t n_ghost_offset = Nx * Ny * radius;
    const index_t n_local = Nx * Ny * Nz;

    // Veven -> input array on even steps, output array on uneven steps.
    // Vodd  -> output array on even steps, input array on uneven steps.
    // Alternation between input and output array allows to implement the stencil as a gather.
    upcxx::dist_object<upcxx::global_ptr<float>> Veven_g = upcxx::new_array<float>(n_local);
    upcxx::dist_object<upcxx::global_ptr<float>> Vodd_g = upcxx::new_array<float>(n_local);    
    float* Veven = downcast_dptr<float>(Veven_g);
    float* Vodd = downcast_dptr<float>(Vodd_g);

    // Vsq, coeff -> coefficients
    upcxx::global_ptr<float> coeff_g = upcxx::new_array<float>(radius+1);
    upcxx::global_ptr<float> Vsq_g = upcxx::new_array<float>(n_local);
    float* coeff = downcast_gptr<float>(coeff_g);
    float* Vsq = downcast_gptr<float>(Vsq_g);

    // Initialize arrays and wait for completion. Veven and Vsq are initialized with pseudo-random
    // numbers; Vodd is left to zero. Coefficients are kept to a fixed value.
    std::mt19937_64 rgen(seed);
    rgen.discard(2 * upcxx::rank_me() * n_block);
    stencil_init_data(Nx, Ny, Nz, radius, rgen, Veven, Vodd, Vsq);

    for (int i = 0; i < radius+1; ++i) {
        coeff[i] = 0.1f;
    }
    upcxx::barrier();

    if (write) {
        dump_stencil(Veven, Vodd, Vsq, n_local, n_ghost_offset, file_path);
    }

    // Begin FDTD
    time_point<Clock> t{};
    if (proc_id == 0 && bench) { // benchmark on the first process
        t = Clock::now();
    }
    // if (bench) {
    //     time_point<Clock> t{};
    //     if (proc_id == 0) {
    //         t = Clock::now();
    //     }
    // TODO: allow variable number of iterations, take average
    //     for (int it = 0; it < iterations; ++it) {
    //     }
    // }
    for (int t = 0; t < steps; ++t) {
        bool is_even_ts = (t & 1) == 0;

        if (proc_n > 1) {
            // std::fprintf(stderr, "Retrieving ghost cells for %s, rank (%d/%d), step %d\n", "Veven", proc_id, proc_n, t);
            stencil_get_ghost_cells(is_even_ts ? Veven_g : Vodd_g,
                                    n_local, n_ghost_offset);
        } // barrier
        stencil_parallel_step(radius, radius + dim_x,
                              radius, radius + dim_y,
                              radius, radius + dim_zi,
                              Nx, Ny, Nz, coeff, Vsq,
                              is_even_ts ? Veven : Vodd,
                              is_even_ts ? Vodd : Veven, 
                              radius);

        // Wait until all processes have finished calculations before proceeding to next step
        upcxx::barrier();
    }

    if (proc_id == 0 && bench) {
        Duration d = Clock::now() -t;
        double time = d.count(); // time in seconds
        // double throughput = dim_x * dim_y * dim_z * sizeof(float) * steps * 1e-9 / time; // throughput in Gb/s
        double throughput = steps * sizeof(float) * (dim_x * dim_y * dim_z * (2 + 2*3*radius) - 2*radius*(dim_x * dim_y + dim_y * dim_z + dim_x * dim_z)) * 1e-9 / time;
        std::fprintf(stdout, "%ld,%ld,%ld,%d,%d,%.12f,%.12f\n", dim_x, dim_y, dim_z, steps, radius, time, throughput);
    }
    if (write) {
        dump_stencil(Veven, Vodd, Vsq, n_local, n_ghost_offset, file_path_steps_cell, true);
        dump_stencil(Veven, Vodd, Vsq, n_local, n_ghost_offset, file_path_steps, false);
    }
    upcxx::finalize();
    // END PARALLEL REGION
}