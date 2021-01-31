#include <iostream>
#include <random>
#include <cassert>
#include <utility>
#include <string>
#include <fstream>
#include <chrono>

#include <cstdlib>
#include <getopt.h>
#include <upcxx/upcxx.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;
template <typename T>
using timePoint = std::chrono::time_point<T>;


template <typename T>
std::ostream& dump_array(std::ostream& stream, T array[], long n) {
    if (stream) {
        for (long i = 0; i < n - 1; ++i) {
            stream << array[i] << " ";
        }
        stream << array[n - 1];
    }
    return stream;
}

template <typename T>
void dump_array_in_rank_order(std::ostream& stream, T array[], long n, const char* label) {
    for (int k = 0; k < upcxx::rank_n(); ++k) {
        if (upcxx::rank_me() == k) {
            if (k == 0) {
                stream << label;
            } else {
                stream << " ";
            }
            dump_array(stream, array, n);
            stream << std::flush; // avoid mangling output
        }
        upcxx::barrier();
    }
}

int main(int argc, char **argv)
{
    long dim = 0;  // amount of rows/columns
    int seed = 42; // seed for pseudo-random generator
    bool write = false;
    bool bench = false;
    const char* file_path = "upcxx_matrix.txt";
    const char* file_path_sym = "upcxx_matrix_symmetrized.txt";

    struct option long_options[] = {
        { "dim", required_argument, NULL, 'd' },
        { "seed", required_argument, NULL, 't' },
        { "write", no_argument, NULL, 'w' },
        { "bench", no_argument, NULL, 'b' },
        { NULL, 0, NULL, 0 }
    };

    int c;
    while ((c = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
        switch (c) {
        case 'd':
            dim = std::stol(optarg);
            break;
        case 't':
            seed = std::stoi(optarg);
            break;
        case 'w':
            write = true;
            break;
        case 'b':
            bench = true;
            break;
        case '?':
            break;
        default:
            std::terminate();
        }
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
    const long N = dim * (dim - 1) / 2;
    const long triag_size = N / nproc;
    assert(triag_size % 2 == 0);
    assert(N == triag_size * nproc);

    const long diag_size = dim / nproc;
    assert(dim == diag_size * nproc);

    // For symmetrization of a square matrix, we consider three arrays:
    // - one holding the lower triangle, in col-major order;
    // - one holding the upper triangle, in row-major order;
    // - one holding the diagonal.
    //
    // Symmetrization does not modify the diagonal, so it could be left out.
    upcxx::global_ptr<float> lower_g(upcxx::new_array<float>(triag_size));
    upcxx::global_ptr<float> upper_g(upcxx::new_array<float>(triag_size));
    upcxx::global_ptr<float> diag_g(upcxx::new_array<float>(diag_size));

    // dist_object: each local value can be accessed with operator* or operator->, but
    // without guarantee that every local value is constructed after the call.
    upcxx::barrier();

    // Downcast to raw pointers, ensuring affinity to the local process
    assert(lower_g.is_local());
    float *lower = lower_g.local();
    assert(upper_g.is_local());
    float *upper = upper_g.local();
    assert(diag_g.is_local());
    float *diag = diag_g.local();

    // Initialize upper and lower triangle with random values
    std::mt19937_64 rgen(seed);
    rgen.discard(proc_id * triag_size * 2); // XXX: offset for pseudo-random generator

    for (long i = 0; i < triag_size; ++i) {
        lower[i] = 0.5 + rgen() % 100;
        upper[i] = 1.0 + rgen() % 100;
    }
    
    // Initialize diagonal (optional)
    long offset_diag = proc_id * diag_size; // offset for diagonal
    for (long i = 0; i < diag_size; ++i) {
        diag[i] = offset_diag + i + 1;
    }
    upcxx::barrier();

    if (write) {
        std::ofstream ofs(file_path, std::ofstream::app);
        if (proc_id == 0) {
            ofs << "DIM: " << dim << "x" << dim << std::endl;
        };
        dump_array_in_rank_order(ofs, lower, triag_size, "LOWER: ");
        dump_array_in_rank_order(ofs, diag, diag_size, "DIAG: ");
        dump_array_in_rank_order(ofs, upper, triag_size, "UPPER: ");
    }

    timePoint<Clock> t;
    if (bench && proc_id == 0) { // measure on single thread
        t = Clock::now();
    }

    // Symmetrize matrix (SAXPY over lower and upper triangle). We only require 
    // a single for loop because lower and upper triangle are stored symmetrically 
    // (in col-major and row-major, respectively)
    for (long i = 0; i < triag_size; ++i) {
        float s = (lower[i] + upper[i]) / 2.;
        lower[i] = s;
        upper[i] = s;
    }
    upcxx::barrier(); // ensure symmetrization is complete
    
    if (bench && proc_id == 0) {
        Duration d = Clock::now() - t;
        double time = d.count(); // time in seconds
        std::cout << time << std::endl;
    }

    // XXX: print dimension
    if (write) {
        std::ofstream ofs(file_path_sym, std::ofstream::app);
        if (proc_id == 0) {
            ofs << "DIM: " << dim << "x" << dim << std::endl;
        };
        dump_array_in_rank_order(ofs, lower, triag_size, "LOWER (C-m): ");
        dump_array_in_rank_order(ofs, diag, diag_size, "DIAG: ");
        dump_array_in_rank_order(ofs, upper, triag_size, "UPPER (R-m): ");
    }

    upcxx::delete_array(lower_g); // XXX: proper way to use with dist_object?
    upcxx::delete_array(upper_g);
    upcxx::delete_array(diag_g);

    upcxx::finalize();
    // END PARALLEL REGION
}