
#include <random>
#include <iostream>
#include <string>
#include <fstream>

#include <cstdlib>
#include <cassert>
#include <getopt.h>
#include <omp.h>

template <typename T>
std::ostream& dump_array(std::ostream& stream, T v[], long n, const char* label) {
    if (stream) {
        stream << label;
        for (long i = 0; i < n - 1; ++i) {
            stream << v[i] << " ";
        }
        stream << v[n - 1] << std::endl;
    }
    return stream;
}

int main(int argc, char** argv) {
    long dim = 0;   // amount of rows/columns
    int seed = 42;  // seed for pseudo-random generator
    bool write = false;

    struct option long_options[] = {
        { "dim",  required_argument, NULL, 'd' },
        { "seed", optional_argument, NULL, 't' },
        { "write", optional_argument, NULL, 'w'},
        { NULL, 0, NULL, 0 }
    };

    int c;
    while ((c = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
        switch(c) {
            case 'd':
                dim = std::stol(optarg);
                break;
            case 't':
                seed = std::stoi(optarg);
                break;
            case 'w':
                write = true;
                break;
            case '?':
                break;
            default:
                std::terminate();
        }
    }
    if (dim <= 0) {
        std::cerr << "a positive dimension is required (specify with --dim)" << std::endl;
        std::exit(1);
    }
    const long triag_size = dim*(dim - 1) / 2;
    const long diag_size = dim;
    
    // For symmetrization of a square matrix, we consider three arrays:
    // - one holding the lower triangle, in col-major order;
    // - one holding the upper triangle, in row-major order;
    // - one holding the diagonal.
    //
    // Symmetrization does not modify the diagonal, so it could be left out.
    float* lower = new float[triag_size];
    float* upper = new float[triag_size];
    float* diag = new float[diag_size];

    // Initialize pseudo-random number generator
    std::mt19937_64 rgen(seed);

#pragma omp parallel firstprivate(rgen)
{
    int nproc = omp_get_num_threads();
    int proc_id = omp_get_thread_num();
    
    long block_size = triag_size / nproc;
    assert(triag_size == nproc * block_size);
    rgen.discard(2 * block_size * proc_id); // advance pseudo-random number generator

#pragma omp for schedule(static)
    for (long i = 0; i < triag_size; ++i)
    {
        lower[i] = 0.5 + rgen() % 100;
        upper[i] = 1.0 + rgen() % 100;
    } // barrier
#pragma omp for schedule(static)
    for (long i = 0; i < diag_size; ++i)
    {
        diag[i] = i + 1;
    } // barrier

#pragma omp single
    if (write) {
        if (std::ofstream stream{"matrix.txt"}; stream) {
            stream << "DIM: " << dim << "x" << dim << std::endl;
            dump_array(stream, lower, triag_size, "LOWER (C-m): ");
            dump_array(stream, diag, diag_size, "DIAG: ");
            dump_array(stream, upper, triag_size, "UPPER (R-m): ");
        }
    }

    // Because lower and upper triangle and stored symmetricaly, we can symmetrize
    // the matrix as a SAXPY operation (over the lower and upper triangle) using a
    // single for loop.
#pragma omp for simd schedule(static)
    for (long i = 0; i < triag_size; ++i)
    {
        double s = (lower[i] + upper[i]) / 2;
        lower[i] = s;
        upper[i] = s;
    }
}
// END PARALLEL REGION

    if (write) {
        if (std::ofstream stream{"matrix_symmetrized.txt"}; stream) {
            stream << "DIM: " << dim << "x" << dim << std::endl;
            dump_array(stream, lower, triag_size, "LOWER (C-m): ");
            dump_array(stream, diag, diag_size, "DIAG: ");
            dump_array(stream, upper, triag_size, "UPPER (R-m): ");
        }
    }
    
    delete[] lower;
    delete[] upper;
    delete[] diag;
}