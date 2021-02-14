#include <random>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include <getopt.h>

#include "include/upcxx.hpp"
#include "include/upcxx_stencil.hpp"
#include "include/upcxx_print.hpp"

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
    bool write = true;
    const char* file_path = "upcxx_stencil.txt";

    index_t dim_x = 8;
    index_t dim_y = 8;
    index_t dim_z = 8;
    int radius = 2;
    int steps = 2;

    // TODO: process options with Lyra

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

    // Serialize E and H arrays (coeff and perm are static and thus left out)
    if (write) {
        dump_stencil(Veven, Vodd, Vsq, n_local, n_ghost_offset, file_path);
    }

    // Begin FDTD
    time_point<Clock> t{};
    if (proc_id == 0 && bench) { // benchmark on the first process
        t = Clock::now();
    }
    // for (int t = 0; t < steps; ++t) {
    //     if ((t & 1) == 0) {
    //         // even step -> update H from E
    //         if (proc_n > 1) {
    //             stencil_get_ghost_cells(Veven_g, n_block, n_ghost_offset);
    //         }
    //         stencil_compute_step(E, H, coeff, perm, dim_x, dim_y, dim_z, n_ghost_offset, n_z, radius);
    //     } else { 
    //         // uneven step -> update E from H
    //         if (proc_n > 1) {
    //             stencil_get_ghost_cells(Vodd_g, n_block, n_ghost_offset);
    //         }
    //         stencil_compute_step(H, E, coeff, perm, dim_x, dim_y, dim_z, n_ghost_offset, n_z, radius);
    //     }
    //     // wait until all processes have finished calculations before proceeding to next step
    //     upcxx::barrier();
    // }
    // if (proc_id == 0 && bench) {
    //     Duration d = Clock::now() -t;
    //     double time = d.count(); // time in seconds
    //     std::fprintf(stdout, "%f.12", time);
    // }
    // if (write) {
    //     dump_stencil(E, H, dim_x, dim_y, dim_z, n_local, n_ghost_offset, file_path);
    // }
    upcxx::finalize();
    // END PARALLEL REGION
}