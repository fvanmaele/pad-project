/*
   Copyright (c) 2010-2011, Intel Corporation
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 * Neither the name of Intel Corporation nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.


 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef STENCIL_HPP
#define STENCIL_HPP
#include <algorithm> // for min()
#include <iostream>
#include <random>

inline void
stencil_init_data(int Nx, int Ny, int Nz,
                  int radius, std::mt19937_64 &rgen,
                  float *Veven, float *Vodd, float *Vsq)
{
    // Current position when iterating over the (3-dimensional) array
    int offset = 0;
    std::uniform_real_distribution<float> dist1(0.0, 1.0);
    std::uniform_real_distribution<float> dist2(0.0, 0.2);

    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x, ++offset)
            {
                // Fill inside of block with pseudo-random values
                if (x >= radius && x < Nx - radius &&
                    y >= radius && y < Ny - radius &&
                    z >= radius && z < Nz - radius)
                {
                    Veven[offset] = dist1(rgen);
                    Vodd[offset] = 0; // NOTE: already intialized by upcxx::new_array/std::vector to 0
                    Vsq[offset] = dist2(rgen);
                }
            }
}

inline void
stencil_parallel_step(int x0,
                      int x1,
                      int y0,
                      int y1,
                      int z0,
                      int z1,
                      int Nx,
                      int Ny,
                      int Nz,
                      const float coeff[],
                      const float vsq[],
                      const float Vin[],
                      float Vout[],
                      const int radius) {
    int Nxy = Nx * Ny;
    auto ind3 = [Nx, Nxy](const int x, const int y, const int z) { 
        return (z * Nxy) + (y * Nx) + x;
        };
    
    for (int z = z0; z < z1; ++z) {
        for (int y = y0; y < y1; ++y) {
#pragma omp simd
            for (int x = x0; x < x1; ++x) {
                auto index = ind3(x, y, z);
                const float *VIin = Vin + index;
                float *VIout = Vout + index;
                float div = coeff[0] * VIin[ind3(0, 0, 0)];

                for (int ir = 1; ir <= radius; ++ir) {
                    div += coeff[ir] * (VIin[ind3(+ir, 0, 0)] + VIin[ind3(-ir, 0, 0)]);
                    div += coeff[ir] * (VIin[ind3(0, +ir, 0)] + VIin[ind3(0, -ir, 0)]);
                    div += coeff[ir] * (VIin[ind3(0, 0, +ir)] + VIin[ind3(0, 0, -ir)]);
                }
                float tmp = 2 * VIin[ind3(0, 0, 0)] - VIout[ind3(0, 0, 0)] + vsq[index] * div;
                VIout[ind3(0, 0, 0)] = tmp;
            }
        }
    }
}

inline void
loop_stencil_parallel(int t0,
                      int t1,
                      int x0,
                      int x1,
                      int y0,
                      int y1,
                      int z0,
                      int z1,
                      int Nx,
                      int Ny,
                      int Nz,
                      const float coeff[],
                      const float vsq[],
                      float Veven[],
                      float Vodd[],
                      const int xtilesize,
                      const int ytilesize,
                      const int ztilesize,
                      const int radius) {
    int cx = 0, cy = 0, cz = 0;
    
    for (int t = t0; t < t1; ++t) {
#pragma omp parallel for collapse(2) schedule(guided)
        for (int z = z0; z < z1; z += ztilesize) {
            for (int y = y0; y < y1; y += ytilesize) {
                for (int x = x0; x < x1; x += xtilesize) {
                    stencil_parallel_step(x, std::min(x1, x + xtilesize), y,
                                          std::min(y1, y + ytilesize), z,
                                          std::min(z1, z + ztilesize), Nx, Ny, Nz,
                                          coeff, vsq, 
                                          (t & 1) == 0 ? Veven : Vodd, 
                                          (t & 1) == 0 ? Vodd : Veven, 
                                          radius);
                }
            }
        }
    }
}

#endif // STENCIL_HPP