#include <iostream>
#include <random>
#include <string>
#include <chrono>
#include <vector>
#include <fstream>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>

//#include <lyra/lyra.hpp>

#include "include/stencil.hpp"

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename T>
using time_point = std::chrono::time_point<T>;
using index_t = std::ptrdiff_t;

template <typename T>
std::ostream &dump_vector(std::ostream &stream, const std::vector<T> &v,
                          const char *label)
{
    size_t n = v.size();
    if (stream) {
        stream << label;
        for (size_t i = 0; i < n - 1; ++i) {
            stream << v[i] << " ";
        }
        stream << v[n - 1] << std::endl;
    }
    return stream;
}

template <typename ...Ns>
bool is_positive(Ns... args) {
    return ((args <= 0) && ...);
}

template <typename T = float>
void init_data(index_t Nx, index_t Ny, index_t Nz, int radius, int seed,
              std::vector<T> &Veven, std::vector<T> &Vodd, std::vector<T> &Vsq)
{
    std::mt19937_64 rgen(seed);

    // Current position when iterating over the (3-dimensional) array
	index_t offset = 0;
    std::uniform_real_distribution<T> dist1(0.0, 1.0);
    std::uniform_real_distribution<T> dist2(0.0, 0.2);

    for (index_t z = 0; z < Nz; ++z)
		for (index_t y = 0; y < Ny; ++y)
			for (index_t x = 0; x < Nx; ++x, ++offset) {
                // Fill inside of block with pseudo-random values
				if(x >= radius && x < Nx - radius &&
				   y >= radius && y < Ny - radius &&
				   z >= radius && z < Nz - radius) 
                {
					//Veven[offset] = (x < Nx / 2) ? x / float(Nx) : y / float(Ny);
                    Veven[offset] = dist1(rgen);
					Vodd[offset] = 0;
					//Vsq[offset] = x * y * z / float(Nx * Ny * Nz);
                    Vsq[offset] = dist2(rgen);
				}
			}
}

int main(int argc, char** argv) 
{
    int seed = 42;  // seed for pseudo-random generator
    bool bench = false;
    bool write = true;
    const char* file_path = "serial_stencil.txt";
    const char* file_path_steps = "serial_stencil_steps.txt";

    // TODO: add Lyra options
    index_t dim_x = 4;
    index_t dim_y = 4;
    index_t dim_z = 4;
    int radius = 1;
    int steps = 1;

    // Array padding, used for accessing neighbors on domain border.
    index_t Nx = dim_x + 2*radius;
    index_t Ny = dim_y + 2*radius;
    index_t Nz = dim_z + 2*radius;
    index_t N = Nx * Ny * Nz;

    // FDTD
    std::mt19937_64 rgen(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    std::vector<float> Veven(N);
    std::vector<float> Vodd(N);
    std::vector<float> Vsq(N);
    std::vector<float> coeff(radius+1);

    // Initialize elements with pseudo-random elements
    init_data<float>(Nx, Ny, Nz, radius, seed, Veven, Vodd, Vsq);
    for (auto&& elem : coeff) {
        elem = 0.1f;
    }

    if (write) {
        std::ofstream stream(file_path, std::ofstream::trunc);
        if (stream) {
            dump_vector(stream, Veven, "Veven: ");
            dump_vector(stream, Vodd, "Vodd: ");
            dump_vector(stream, Vsq, "Vsq: ");
        }
    }

    for (int t = 0; t < steps; ++t) {
        stencil_parallel_step(radius, radius + dim_x, radius, radius + dim_y, radius, radius + dim_z,
                              Nx, Ny, Nz, coeff.data(), Vsq.data(),
                              ((t&1) == 0 ? Veven.data() : Vodd.data()), 
                              ((t&1) == 0 ? Vodd.data() : Veven.data()), 
                              radius);
    }

    if (write) {
        std::ofstream stream(file_path_steps, std::ofstream::trunc);
        if (stream) {
            dump_vector(stream, Veven, "Veven: ");
            dump_vector(stream, Vodd, "Vodd: ");
            dump_vector(stream, Vsq, "Vsq: ");
        }
    }

}