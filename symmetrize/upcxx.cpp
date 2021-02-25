
#include <random>
#include <iostream>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <utility>
#include <string>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <filesystem>

#include <upcxx/upcxx.hpp>
#include <lyra/lyra.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename T>
using time_point = std::chrono::time_point<T>;
using index_t = std::ptrdiff_t;

template <typename T>
std::ostream& 
dump_vector(std::ostream& stream, const std::vector<T> &vec, index_t n) {
    if (stream) {
        for (index_t i = 0; i < n - 1; ++i) {
            stream << vec[i] << " ";
        }
        stream << vec[n - 1];
    }
    return stream;
}

template <typename T>
void dump_vector_in_rank_order(std::ostream &stream, const std::vector<T> &vec, index_t n,
                               const char *label) {
    for (int k = 0; k < upcxx::rank_n(); ++k) {
        if (upcxx::rank_me() == k) {
            if (k == 0) {
                stream << label;
            } else {
                stream << " ";
            }
            dump_vector(stream, vec, n);
            stream << std::flush; // avoid mangling output
            if (k == upcxx::rank_n() - 1) {
                stream << std::endl;
            }
        }
        upcxx::barrier();
    }
}


int main(int argc, char **argv)
{
    index_t dim = 0;  // amount of rows/columns
    int seed = 42; // seed for pseudo-random generator
    int iterations = 1;
    bool write = false;
    bool bench = false;
    bool show_help = false;
    std::filesystem::path file_path("upcxx_matrix.txt");
    std::filesystem::path file_path_sym("upcxx_matrix_symmetrized.txt");

    auto cli = lyra::help(show_help) |
        lyra::opt(dim, "dim")["-N"]["--dim"](
            "Size of reduced vec, must be specified") |
        lyra::opt(iterations, "iterations")["--iterations"](
            "Number of iterations, default is 1") |
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
    
    // BEGIN PARALLEL REGION
    upcxx::init();
    int nproc = upcxx::rank_n();
    int proc_id = upcxx::rank_me();

    // Block size for each process
    const index_t N = dim * (dim - 1) / 2;
    const index_t triangle_n = N / nproc;
    assert(triangle_n % 2 == 0);
    assert(N == triangle_n * nproc);

    const index_t diagonal_n = dim / nproc;
    assert(dim == diagonal_n * nproc);

    // For symmetrization of a square matrix, we consider three arrays:
    // - one holding the lower triangle, in col-major order;
    // - one holding the upper triangle, in row-major order;
    // - one holding the diagonal.
    //
    // Symmetrization does not modify the diagonal, so it could be left out.
    std::vector<float> lower(triangle_n);
    std::vector<float> upper(triangle_n);
    std::vector<float> diag(diagonal_n);

    // Initialize upper and lower triangle with pseudo-random values
    std::mt19937_64 rgen(seed);
    rgen.discard(proc_id * triangle_n * 2);
    for (index_t i = 0; i < triangle_n; ++i) {
        lower[i] = 0.5 + rgen() % 100;
        upper[i] = 1.0 + rgen() % 100;
    }
    
    // Initialize diagonal (optional)
    index_t offset_diag = proc_id * diagonal_n; // offset for diagonal
    for (index_t i = 0; i < diagonal_n; ++i) {
        diag[i] = offset_diag + i + 1;
    }

    if (write) {
        if (proc_id == 0) {
            std::ofstream ofs(file_path.c_str(), std::ofstream::trunc);
            ofs << "DIM: " << dim << "x" << dim << std::endl;
        };
        upcxx::barrier();
        std::ofstream ofs(file_path.c_str(), std::ofstream::app);

        dump_vector_in_rank_order(ofs, lower, triangle_n, "LOWER (C-m): ");
        dump_vector_in_rank_order(ofs, diag, diagonal_n, "DIAG: ");
        dump_vector_in_rank_order(ofs, upper, triangle_n, "UPPER (R-m): ");
    }

    // Timings for different iterations, of which the mean is taken.
    std::vector<double> vt;
    vt.reserve(iterations);
    
    // Copies for multiple iterations (in-place transposition)
    std::vector<float> lower_cp(triangle_n);
    std::vector<float> upper_cp(triangle_n);
    
    // Symmetrization
    for (int iter = 1; iter <= iterations; ++iter) {
        std::copy(lower.begin(), lower.end(), lower_cp.begin());
        std::copy(upper.begin(), upper.end(), upper_cp.begin());

        // Set up a barrier before doing any timing
        upcxx::barrier();
        time_point<Clock> t = Clock::now();

        // Symmetrize matrix (SAXPY over lower and upper triangle). We only require 
        // a single for loop because lower and upper triangle are stored symmetrically 
        // (in col-major and row-major, respectively)
        for (index_t i = 0; i < triangle_n; ++i) {
            float s = (lower_cp[i] + upper_cp[i]) / 2.;
            lower_cp[i] = s;
            upper_cp[i] = s;
        }
        upcxx::barrier(); // ensure symmetrization is complete
        
        if (proc_id == 0) {
            Duration d = Clock::now() - t;
            double time = d.count(); // time in seconds
            vt.push_back(time);
        }
    }
    if (proc_id == 0) {
        double time = std::accumulate(vt.begin(), vt.end(), 0.);
        time /= iterations;
        
        double throughput = dim * (dim-1) * sizeof(float) * 1e-9 / time;
        std::fprintf(stdout, "%ld,%.12f,%.12f\n", dim, time, throughput);
    }
    
    if (write) {
        if (proc_id == 0) {
            std::ofstream ofs(file_path_sym.c_str(), std::ofstream::trunc);
            ofs << "DIM: " << dim << "x" << dim << std::endl;
        };
        upcxx::barrier();
        std::ofstream ofs(file_path_sym.c_str(), std::ofstream::app);

        dump_vector_in_rank_order(ofs, lower_cp, triangle_n, "LOWER (C-m): ");
        dump_vector_in_rank_order(ofs, diag, diagonal_n, "DIAG: ");
        dump_vector_in_rank_order(ofs, upper_cp, triangle_n, "UPPER (R-m): ");
    }
    
    upcxx::finalize();
    // END PARALLEL REGION
}
