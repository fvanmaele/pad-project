
#include <random>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include <cstdlib>
#include <getopt.h>

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
    const long triangle_size = dim*(dim - 1) / 2;
    const long diag_size = dim;
    
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
    for (long i = 0; i < triangle_size; ++i) {
        lower[i] = 0.5 + rgen() % 100;
        upper[i] = 1.0 + rgen() % 100;
    }
    for (long i = 0; i < diag_size; ++i) {
        diag[i] = i + 1;
    }

    // Seralize original matrix
    if (write) {
        if (std::ofstream stream{"matrix.txt"}; stream) {
            stream << "DIM: " << dim << "x" << dim << std::endl;
            dump_vector(stream, lower, "LOWER (C-m): ");
            dump_vector(stream, diag, "DIAG: ");
            dump_vector(stream, upper, "UPPER (R-m): ");
        }
    }

    // Because lower and upper triangle and stored symmetricaly, we can symmetrize
    // the matrix as a SAXPY operation (over the lower and upper triangle) in a
    // single for loop.
    for (long i = 0; i < triangle_size; ++i) {
        double s = (lower[i] + upper[i]) / 2;
        lower[i] = s;
        upper[i] = s;
    }

    // Serialize new (symmetrized) matrix
    if (write) {
        if (std::ofstream stream{"matrix_symmetrized.txt"}; stream) {
            stream << "DIM: " << dim << "x" << dim << std::endl;
            dump_vector(stream, lower, "LOWER (C-m): ");
            dump_vector(stream, diag, "DIAG: ");
            dump_vector(stream, upper, "UPPER (R-m): ");
        }
    }
}