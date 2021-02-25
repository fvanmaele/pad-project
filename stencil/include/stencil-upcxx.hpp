#ifndef UPCXX_STENCIL_HPP
#define UPCXX_STENCIL_HPP
#include <cassert>
#include "upcxx.hpp"

inline void
stencil_get_ghost_cells(dist_ptr<float> &input_g, index_t n_local, index_t n_ghost_offset)
{
    // XXX: use upcxx::local_team() to reduce overhead when accessing elements on the same node
    const upcxx::intrank_t proc_n = upcxx::rank_n();
    const upcxx::intrank_t proc_id = upcxx::rank_me();
    assert(proc_n > 1);

    // Downcast to regular C++ pointer
    float *input = downcast_dptr<float>(input_g);

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
        upcxx::rget(input_l + n_local - 2*n_ghost_offset,
                    input,
                    n_ghost_offset).wait();
    }
    upcxx::barrier();
}

#endif // UPCXX_STENCIL_HPP
