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

int main(int argc, char** argv) 
{
    int seed = 42;  // seed for pseudo-random generator
    bool bench = false;
    bool write = true;
    const char* file_path = "serial_stencil.txt";
    const char* file_path_steps = "serial_stencil_steps.txt";

    // TODO: add Lyra options
    index_t dim_x = 32;
    index_t dim_y = 32;
    index_t dim_z = 32;
    int radius = 4;
    int steps = 5;

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
    stencil_init_data(Nx, Ny, Nz, radius, rgen, Veven.data(), Vodd.data(), Vsq.data());
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