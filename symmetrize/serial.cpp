
#include <random>
#include <iostream>
#include <cstddef>
#include <cstdio>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>

#include <cstdlib>
#include <lyra/lyra.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename T>
using timePoint = std::chrono::time_point<T>;
using index_t = std::ptrdiff_t;

template <typename T>
std::ostream& dump_vector(std::ostream& stream, const std::vector<T>& v, const char* label) {
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

int main(int argc, char** argv) {
    index_t dim = 0;   // amount of rows/columns
    int seed = 42;  // seed for pseudo-random generator
    bool bench = false;
    bool write = false;
    bool show_help = false;
    std::filesystem::path file_path("serial_matrix.txt");
    std::filesystem::path file_path_sym("serial_matrix_symmetrized.txt");

    auto cli = lyra::help(show_help) |
        lyra::opt(dim, "dim")["-N"]["--dim"](
            "Size of reduced vec, must be specified") |
        lyra::opt(seed, "seed")["--seed"](
            "Seed for pseudo-random number generation, default is 42") |
        lyra::opt(bench)["--bench"](
            "Print benchmarks to standard output") |
        lyra::opt(write)["--write"](
            "Serialize matrix before and after symmetrization");
    auto result = cli.parse({argc, argv});

    if (!result) {
		std::cerr << "Error in command line: " << result.errorMessage()
			  << std::endl;
		exit(1);
	}
	if (show_help) {
		std::cout << cli << std::endl;
		exit(0);
	}
    if (dim <= 0) {
        std::cerr << "positive dimension required (specify with --dim)" << std::endl;
        std::exit(1);
    }
    const index_t triangle_size = dim*(dim - 1) / 2;
    const index_t diag_size = dim;
    
    // For symmetrization of a square matrix, we consider three arrays:
    // - one holding the lower triangle, in col-major order;
    // - one holding the upper triangle, in row-major order;
    // - one holding the diagonal.
    //
    // Symmetrization does not modify the diagonal, so it could be left out.
    std::vector<float> lower(triangle_size);
    std::vector<float> upper(triangle_size);
    std::vector<float> diag(diag_size);
    
    std::mt19937_64 rgen(seed);
    for (index_t i = 0; i < triangle_size; ++i) {
        lower[i] = 0.5 + rgen() % 100;
        upper[i] = 1.0 + rgen() % 100;
    }
    for (index_t i = 0; i < diag_size; ++i) {
        diag[i] = i + 1;
    }

    // Seralize original matrix
    if (write) {
        std::ofstream stream{file_path.c_str()};

        if (stream) {
            stream << "DIM: " << dim << "x" << dim << std::endl;
            dump_vector(stream, lower, "LOWER (C-m): ");
            dump_vector(stream, diag, "DIAG: ");
            dump_vector(stream, upper, "UPPER (R-m): ");
        }
    }

    // XXX: implement multiple iterations as in parallel implementation (e.g. to measure speedup)
    timePoint<Clock> t = Clock::now();

    // Because lower and upper triangle and stored symmetricaly, we can symmetrize
    // the matrix as a SAXPY operation (over the lower and upper triangle) in a
    // single for loop.
    for (index_t i = 0; i < triangle_size; ++i) {
        double s = (lower[i] + upper[i]) / 2;
        lower[i] = s;
        upper[i] = s;
    }

    Duration d = Clock::now() - t;
    double time = d.count(); // time in seconds
    double throughput = dim * (dim-1) * sizeof(float) * 1e-9 / time;
    std::fprintf(stdout, "%ld,%.12f,%.12f\n", dim, time, throughput);

    // Serialize new (symmetrized) matrix
    if (write) {
        std::ofstream stream{file_path_sym.c_str()};

        if (stream) {
            stream << "DIM: " << dim << "x" << dim << std::endl;
            dump_vector(stream, lower, "LOWER (C-m): ");
            dump_vector(stream, diag, "DIAG: ");
            dump_vector(stream, upper, "UPPER (R-m): ");
        }
    }
}