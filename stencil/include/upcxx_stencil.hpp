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
			for (index_t x = 0; x < Nx; ++x, ++offset) {
                // Fill inside of block with pseudo-random values
				if(x >= radius && x < Nx - radius &&
				   y >= radius && y < Ny - radius &&
				   z >= radius && z < Nz - radius) 
                {
                    Veven[offset] = dist1(rgen);
					//Vodd[offset] = 0; // already set by upcxx::new_array
                    Vsq[offset] = dist2(rgen);
				}
			}
}

inline void 
stencil_get_ghost_cells(dist_ptr<float> &input_g, index_t n_block, index_t n_ghost_offset)
{
    // XXX: use upcxx::local_team() to reduce overhead when accessing elements on the same node
    const upcxx::intrank_t proc_n = upcxx::rank_n();
    const upcxx::intrank_t proc_id = upcxx::rank_me();
    assert(proc_n > 1);

    // Downcast to regular C++ pointer
    float* input = downcast_dptr<float>(input_g);

    // Fetch the left and right pointers for the ghost cells.
    // NOTE: We define neighbors to be periodic, i.e. process 0 has process n-1 as left neighbor,
    // and process n-1 has process 0 as right neighbor.
    upcxx::intrank_t l_nbr = (proc_id + proc_n - 1) % proc_n;
    upcxx::intrank_t r_nbr = (proc_id + 1) % proc_n;
    upcxx::global_ptr<float> input_L;
    upcxx::global_ptr<float> input_R;

    // XXX: Because the fetch function is asynchronous, we have to synchronize on completion,
    // using a call to wait(). Later, we will see how to overlap asynchronous operations, that
    // is, when communication is split-phased.

    // XXX: this should only be done if there at least two processes
    if (proc_id == 0) {
        // Lower ghost cells are zero (lower boundary of domain), fetch of left nbr not required
        input_R = input_g.fetch(r_nbr).wait();
        upcxx::rget(input_R + n_ghost_offset, input + n_ghost_offset + n_block, n_ghost_offset).wait();
    } 
    else if (proc_id == proc_n - 1) {
        // Upper ghost cells are zero (upper boundary of domain), fetch of right nbr not required
        input_L = input_g.fetch(l_nbr).wait();
        upcxx::rget(input_L + n_block, input, n_ghost_offset).wait();
    } 
    else {
        // Retrieve both lower and upper ghost cells from neighbors
        input_L = input_g.fetch(l_nbr).wait();
        upcxx::rget(input_L + n_block, input, n_ghost_offset).wait();
        input_R = input_g.fetch(r_nbr).wait();
        upcxx::rget(input_R + n_ghost_offset, input + n_ghost_offset + n_block, n_ghost_offset).wait();
    }
}

inline void
stencil_compute_step(float *input, float *output, float *coeff, float *perm,
                     index_t dim_x, index_t dim_y, index_t dim_z,
                     index_t n_ghost_offset, index_t n_z, int radius)
{
    // 3-dimensional arrays are accessed through offsets on a regular 1-dimensional array.
    auto ind3 = [dim_x, dim_y](index_t x, index_t y, index_t z) { 
        return (z * dim_x * dim_y) + (y * dim_x) + x;
    };

    // Compute values for time step
    const upcxx::intrank_t proc_id = upcxx::rank_me();
    index_t z0 = n_z * proc_id; // subdivision in z direction
    index_t z1 = n_z * (proc_id+1);

    // XXX: as the provided template function (FDTD3d/stencil-parallel.h), this assumes zero padding
    // on x, y and z borders (use of ind3 as index function). In the upcxx implementation, "padding"
    // for the z-direction is provided for ghost cells, but no such padding is done for x or y.
    for (index_t z = z0; z < z1; ++z) {
        assert(z < dim_z);
        for (index_t x = 0; x < dim_x; ++x) {
            for (index_t y = 0; y < dim_y; ++y) {
                index_t index = ind3(x, y, z);
                // inside block of stencil (input array, first element)
                const float* input_blk = input + n_ghost_offset;
                // inside block of stencil (input array, computed element)
                const float* input_blk_p = input_blk + index;
                // inside block of stencil (output array, first element)
                float* output_blk = output + n_ghost_offset;
                // outside block of stencil (output array, computed element)
                float* output_blk_p = output_blk + index;
                
                // XXX: add some basic boundary checks (for upper and lower blocks, cf. picture)

                // Update divergence
                float div = coeff[0] * input_blk_p[0];
                for (int ir = 1; ir <= radius; ++ir) {
                    div += coeff[ir] * (input_blk_p[ind3(+ir, 0, 0)] + input_blk_p[ind3(-ir, 0, 0)]);
                    div += coeff[ir] * (input_blk_p[ind3(0, +ir, 0)] + input_blk_p[ind3(0, -ir, 0)]);
                    div += coeff[ir] * (input_blk_p[ind3(0, 0, +ir)] + input_blk_p[ind3(0, 0, -ir)]);
                }
                float tmp = 2 * input_blk_p[0] - output_blk_p[0] + perm[index] * div;
                output_blk_p[0] = tmp;
            }
        }
    }
}

#endif // UPCXX_STENCIL_HPP