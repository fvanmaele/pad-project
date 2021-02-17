#include <iostream>
#include <random>
#include <string>
#include <chrono>
#include <vector>
#include <fstream>
#include <cassert>
#include <cstdio>
#include <lyra/lyra.hpp>

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
    return ((args > 0) && ...);
}

int main(int argc, char** argv) 
{
    int seed = 42;  // seed for pseudo-random generator
    bool bench = false;
    bool write = false;
    bool show_help = false;
    const char* file_path = "serial_stencil.txt";
    const char* file_path_steps = "serial_stencil_steps.txt";

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

    // Array padding, used for accessing neighbors on domain border.
    index_t Nx = dim_x + 2*radius;
    index_t Ny = dim_y + 2*radius;
    index_t Nz = dim_z + 2*radius;
    index_t N = Nx * Ny * Nz;

    // FDTD
    std::mt19937_64 rgen(seed);
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

    // Begin FDTD
    time_point<Clock> t{};
    if (bench) {
        t = Clock::now();
    }
    for (int t = 0; t < steps; ++t) {
        stencil_parallel_step(radius, radius + dim_x, radius, radius + dim_y, radius, radius + dim_z,
                              Nx, Ny, Nz, coeff.data(), Vsq.data(),
                              ((t&1) == 0 ? Veven.data() : Vodd.data()), 
                              ((t&1) == 0 ? Vodd.data() : Veven.data()), 
                              radius);
    }
    if (bench) {
        Duration d = Clock::now() -t;
        double time = d.count(); // time in seconds
        double throughput = dim_x * dim_y * dim_z * sizeof(float) * steps * 1e-9 / time; // throughput in Gb/s
        std::fprintf(stdout, "%d,%d,%d,%d,%f.12,%f.12\n", dim_x, dim_y, dim_z, steps, throughput, time);
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