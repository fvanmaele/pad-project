#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>
#include <random>

#include "trimatrix.h"
#include "matrix.h"

using namespace asc::pad_ws20::project;

template <typename Matrix>
using Value = typename Matrix::value_type;

TEST_CASE("TriMatrix to SquareMatrix conversion") {
    double diag[5] = {   // diagonal
        1, 7, 13, 19, 25
    };
    double lower[10] = { // lower triangle (col-major order)
        6, 11, 16, 21, 12, 17, 22, 18, 23, 24
    };
    double upper[10] = { // upper triangle (row-major order)
        2, 3, 4, 5, 8, 9, 10, 14, 15, 20
    };
    double elems[25] = { // elements (row-major order)
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    TriMatrix<double> T1(diag, 5, lower, upper, 10);
    REQUIRE(T1.n() == 5);
    REQUIRE(T1.t() == 10);
    for (int i = 0; i < 5; ++i) {
        CAPTURE(i);
        CHECK(T1.diag()[i] == diag[i]);
    }
    for (int k = 0; k < 10; ++k) {
        CAPTURE(k);
        CHECK(T1.lower()[k] == lower[k]);
        CHECK(T1.upper()[k] == upper[k]);
    }

    TriMatrix<double> T2(elems, 25);
    REQUIRE(T1.n() == 5);
    REQUIRE(T1.t() == 10);
    for (int i = 0; i < 5; ++i) {
        CAPTURE(i);
        CHECK(T1.diag()[i] == diag[i]);
    }
    for (int k = 0; k < 10; ++k) {
        CAPTURE(k);
        CHECK(T1.lower()[k] == lower[k]);
        CHECK(T1.upper()[k] == upper[k]);
    }
    TriMatrix<double> T3(5);
    CHECK(T3.n() == 5);
    CHECK(T3.t() == 10);
    CHECK(T3.s() == 25);
    
    SquareMatrix<double> M1(elems, 25);
    for (int k = 0; k < 25; ++k) {
        CAPTURE(k);
        CHECK(M1.elements()[k] == elems[k]);
    }
    SquareMatrix<double> M2(5);
    CHECK(M2.n() == 5);
    CHECK(M2.t() == 10);
    CHECK(M2.s() == 25);
}

TEMPLATE_TEST_CASE("square matrix", "", TriMatrix<double>, SquareMatrix<double>) {
    using Matrix = TestType;

    double diag[5] = {   // diagonal
        1, 7, 13, 19, 25
    };
    double elems[25] = { // full matrix
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    // Initialization of matrix (redone for each section)
    Matrix M(elems, 25);
    int n = M.n();

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
        Matrix N(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                N(i, j) = M(i, j);
            }
        }
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
                CHECK(M(i, j) == Approx((N(i, j) + N(j, i)) / 2.));
            }
        }
    }
}
