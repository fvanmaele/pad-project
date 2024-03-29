#pragma once

void stencil_parallel_step(int x0,
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
                           const int radius);

void loop_stencil_parallel(int t0,
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
                           const int radius);
