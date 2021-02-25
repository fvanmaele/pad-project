
#include <random>
#include <iostream>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <filesystem>

#include <omp.h>
#include <lyra/lyra.hpp>
#include <upcxx/upcxx.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename T>
using time_point = std::chrono::time_point<T>;
using index_t = std::ptrdiff_t;

template <typename T>
std::ostream& dump_array(std::ostream& stream, T array[], index_t n) {
    if (stream) {
        for (index_t i = 0; i < n - 1; ++i) {
            stream << array[i] << " ";
        }
        stream << array[n - 1];
    }
    return stream;
}

template <typename T>
void dump_array_in_rank_order(std::ostream &stream, T array[], index_t n,
                               const char *label) {
    for (int k = 0; k < upcxx::rank_n(); ++k) {
        if (upcxx::rank_me() == k) {
            if (k == 0) {
                stream << label;
            } else {
                stream << " ";
            }
            dump_array(stream, array, n);
            stream << std::flush; // avoid mangling output
            if (k == upcxx::rank_n() - 1) {
                stream << std::endl;
            }
        }
        upcxx::barrier();
    }
}

int main(int argc, char** argv) {
    index_t dim = 0;   // amount of rows/columns
    int seed = 42;  // seed for pseudo-random generator
    int iterations = 1;
    bool bench = false;
    bool write = false;
    bool show_help = false;
    std::filesystem::path file_path("openmp_matrix.txt");
    std::filesystem::path file_path_sym("openmp_matrix_symmetrized.txt");

    auto cli = lyra::help(show_help) |
        lyra::opt(dim, "dim")["-N"]["--dim"](
            "Size of reduced array, must be specified") |
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
    upcxx::intrank_t nproc = upcxx::rank_n();
    upcxx::intrank_t proc_id = upcxx::rank_me();

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
    float* lower = new float[triangle_n];
    float* upper = new float[triangle_n];
    float* diag = new float[diagonal_n];

    // Initialize pseudo-random number generator
    std::mt19937_64 rgen(seed);

// XXX: integrate with upcxx
#pragma omp parallel firstprivate(rgen)
{
    int threads = omp_get_num_threads();

    index_t block_size = triangle_n / threads;
    assert(triangle_n == threads * block_size);

    rgen.discard(2 * (proc_id * threads + omp_get_thread_num()) * block_size);

#pragma omp for schedule(static)
    for (index_t i = 0; i < triangle_n; ++i) {
        lower[i] = 0.5 + rgen() % 100;
        upper[i] = 1.0 + rgen() % 100;
    } // barrier

    index_t offset_diag = proc_id * diagonal_n;
#pragma omp for schedule(static)
    for (index_t i = 0; i < diagonal_n; ++i) {
        diag[i] = offset_diag + i + 1;
    } // barrier
}

    if (write) {
        if (proc_id == 0) {
            std::ofstream ofs(file_path.c_str(), std::ofstream::trunc);
            ofs << "DIM: " << dim << "x" << dim << std::endl;
        };
        upcxx::barrier();
        std::ofstream ofs(file_path.c_str(), std::ofstream::app);

        dump_array_in_rank_order(ofs, lower, triangle_n, "LOWER (C-m): ");
        dump_array_in_rank_order(ofs, diag, diagonal_n, "DIAG: ");
        dump_array_in_rank_order(ofs, upper, triangle_n, "UPPER (R-m): ");
    }


    // Timings for different iterations, of which the mean is taken.
    std::vector<double> vt;
    vt.reserve(iterations);
    
    // Copies for multiple iterations (in-place transposition)
    float* lower_cp = new float[triangle_n];
    float* upper_cp = new float[triangle_n];

    for (int iter = 1; iter <= iterations; ++iter) {
        // Initialize matrix
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i < triangle_n; ++i) {
            lower_cp[i] = lower[i];
            upper_cp[i] = upper[i];
        }
    
        // Set up a barrier before doing any timing
        upcxx::barrier();
        time_point<Clock> t = Clock::now();

        // Because lower and upper triangle and stored symmetricaly, we can symmetrize
        // the matrix as a SAXPY operation (over the lower and upper triangle) using a
        // single for loop.
#pragma omp parallel for simd schedule(static)
        for (index_t i = 0; i < triangle_n; ++i) {
            double s = (lower_cp[i] + upper_cp[i]) / 2;
            lower_cp[i] = s;
            upper_cp[i] = s;
        }
        upcxx::barrier();

        if (proc_id == 0) {
            Duration d = Clock::now() - t;
            double time = d.count(); // time in seconds
            vt.push_back(time);
        }
    }
      
    if (write) {
        if (proc_id == 0) {
            std::ofstream ofs(file_path_sym.c_str(), std::ofstream::trunc);
            ofs << "DIM: " << dim << "x" << dim << std::endl;
        };
        upcxx::barrier();
        std::ofstream ofs(file_path_sym.c_str(), std::ofstream::app);

        dump_array_in_rank_order(ofs, lower_cp, triangle_n, "LOWER (C-m): ");
        dump_array_in_rank_order(ofs, diag, diagonal_n, "DIAG: ");
        dump_array_in_rank_order(ofs, upper_cp, triangle_n, "UPPER (R-m): ");
    }

    delete[] lower;
    delete[] upper;
    delete[] diag;
    delete[] lower_cp;
    delete[] upper_cp;

    upcxx::finalize();
    // END PARALLEL REGION
}
