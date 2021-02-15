#ifndef UPCXX_STENCIL_HPP
#define UPCXX_STENCIL_HPP
#include <cassert>
#include "upcxx.hpp"

inline void
stencil_init_data(index_t Nx, index_t Ny, index_t Nz,
                  int radius, std::mt19937_64 &rgen,
                  float *Veven, float *Vodd, float *Vsq)
{
    // Current position when iterating over the (3-dimensional) array
    index_t offset = 0;
    std::uniform_real_distribution<float> dist1(0.0, 1.0);
    std::uniform_real_distribution<float> dist2(0.0, 0.2);

    for (index_t z = 0; z < Nz; ++z)
        for (index_t y = 0; y < Ny; ++y)
            for (index_t x = 0; x < Nx; ++x, ++offset)
            {
                // Fill inside of block with pseudo-random values
                if (x >= radius && x < Nx - radius &&
                    y >= radius && y < Ny - radius &&
                    z >= radius && z < Nz - radius)
                {
                    Veven[offset] = dist1(rgen);
                    Vodd[offset] = 0; // NOTE: already intialized by upcxx::new_array to 0
                    Vsq[offset] = dist2(rgen);
                }
            }
}

inline void
stencil_get_ghost_cells(dist_ptr<float> &input_g, index_t n_local, index_t n_ghost_offset)
{
    // XXX: use upcxx::local_team() to reduce overhead when accessing elements on the same node
    const upcxx::intrank_t proc_n = upcxx::rank_n();
    const upcxx::intrank_t proc_id = upcxx::rank_me();
    assert(proc_n > 1);

    // Downcast to regular C++ pointer
    float *input = downcast_dptr<float>(input_g);

    // XXX: Because the fetch function is asynchronous, we have to synchronize on completion,
    // using a call to wait(). Later, we will see how to overlap asynchronous operations, that
    // is, when communication is split-phased.

    // As rget does not allow source values to be modified until operation completion is notified,
    // first retrieve all right neighbors, then all left neighbors.
    if (proc_id != proc_n - 1) {
        upcxx::global_ptr<float> input_r = input_g.fetch(proc_id + 1).wait();
        upcxx::rget(input_r + n_ghost_offset,
                    input + n_local - n_ghost_offset,
                    n_ghost_offset).wait();
    }
    upcxx::barrier();

    if (proc_id != 0) {
        upcxx::global_ptr<float> input_l = input_g.fetch(proc_id - 1).wait();
        upcxx::rget(input_l + n_local - 2 * n_ghost_offset,
                    input,
                    n_ghost_offset).wait();
    }
    upcxx::barrier();

}

#endif // UPCXX_STENCIL_HPP