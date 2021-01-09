#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>

#include "trimatrix.h"
#include "matrix.h"

using namespace PAD;

template <typename Matrix>
Matrix init_matrix(double[], double[], double[], double[],
                   ptrdiff_t, ptrdiff_t, ptrdiff_t) {
    std::terminate();
}

template <>
TriMatrix init_matrix<TriMatrix>(double diag[], double lower[], double upper[], double[],
                                 ptrdiff_t n, ptrdiff_t t, ptrdiff_t) {
    TriMatrix M(n);
    double* M_D = M.diag();
    double* M_L = M.lower();
    double* M_R = M.upper();

    for (ptrdiff_t i = 0; i < n; ++i) {
        M_D[i] = diag[i];
    }
    for (ptrdiff_t i = 0; i < t; ++i) {
        M_L[i] = lower[i];
    }
    for (ptrdiff_t i = 0; i < t; ++i) {
        M_R[i] = upper[i];
    }
    return M;
}

template <>
SquareMatrix init_matrix<SquareMatrix>(double[], double[], double[], double elems[],
                                       ptrdiff_t n, ptrdiff_t, ptrdiff_t s) {
    SquareMatrix M(n);
    double* M_elems = M.elements();

    for (ptrdiff_t k = 0; k < s; ++k) {
        M_elems[k] = elems[k];
    }
    return M;
}

TEMPLATE_TEST_CASE("square matrix", "", TriMatrix, SquareMatrix) {
    using Matrix = TestType;

    REQUIRE_THROWS(Matrix(-1));
    REQUIRE_THROWS(Matrix(0));
    REQUIRE_NOTHROW(Matrix(1));
    REQUIRE_NOTHROW(Matrix(2));

    double diag[5] = {   // diagonal
        1, 7, 13, 19, 25
    };
    double lower[10] = { // lower triangle (col-major order)
        6, 11, 16, 21, 12, 17, 22, 18, 23, 24
    };
    double upper[10] = { // upper triangle (row-major)
        2, 3, 4, 5, 8, 9, 10, 14, 15, 20
    };
    double elems[25] = { // full matrix
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    // Initialization of matrix (redone for each section)
    Matrix M = init_matrix<Matrix>(diag, lower, upper, elems, 5, 10, 25);
    ptrdiff_t n = M.n();
    ptrdiff_t t = M.t();
    ptrdiff_t s = M.s();

    SECTION ("dimension") {
        CHECK(n == 5);
        CHECK(t == 10);
        CHECK(s == 25);
    }

    SECTION("const accessor") {
        int k = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i,j) == k);
                ++k;
            }
        }
    }

    SECTION("non-const accessor") {
        Matrix T(5);
        int k = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                T(i, j) = k;
                ++k;
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i, j) == T(i, j));
            }
        }
    }

    SECTION("transposition") {
        M.transpose();
        int k = 1;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i, j) == k);
                ++k;
            }
        }

        M.transpose();
        k = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i, j) == k);
                ++k;
            }
        }
    }
    
    SECTION("symmetrize matrix") {
        M.symmetrize();

        // double diag[5] = { 1, 7, 13, 19, 25 };
        for (int i = 0; i < n; ++i) {
            CHECK(M(i, i) == diag[i]);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i, j) == M(j, i));
            }
        }
    }
}
