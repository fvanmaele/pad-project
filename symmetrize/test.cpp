#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>

#include "trimatrix.h"

TEST_CASE("TriMatrix") {
    REQUIRE_THROWS(TriMatrix(-1));
    REQUIRE_THROWS(TriMatrix(0));
    REQUIRE_NOTHROW(TriMatrix(1));

    // Initialization of matrix (redone for every SECTION)
    TriMatrix M(5);
    CHECK(M._n == 5);
    CHECK(M._t == 10);
    CHECK(M._s() == 25);

    // Matrix diagonal
    double diag[5] = { 1, 7, 13, 19, 25 };
    // Lower triangle (col-major order)
    double lower[10] = { 6, 11, 16, 21, 12, 17, 22, 18, 23, 24 };
    // Upper triangle (row-major order)
    double upper[10] = { 2, 3, 4, 5, 8, 9, 10, 14, 15, 20 };

    double* D = M.diag();
    for (int i = 0; i < M._n; ++i) {
        D[i] = diag[i];
    }
    double* L = M.lower();
    for (int i = 0; i < M._t; ++i) {
        L[i] = lower[i];
    }
    double* R = M.upper();
    for (int i = 0; i < M._t; ++i) {
        R[i] = upper[i];
    }
    
    SECTION("const accessor") {
        int k = 1;
        for (int i = 0; i < M._n; ++i) {
            for (int j = 0; j < M._n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i,j) == k);
                ++k;
            }
        }
    }

    SECTION("non-const accessor") {
        TriMatrix T(5);
        int k = 1;
        for (int i = 0; i < M._n; ++i) {
            for (int j = 0; j < M._n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                T(i, j) = k;
                ++k;
            }
        }
        for (int i = 0; i < M._n; ++i) {
            for (int j = 0; j < M._n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i, j) == T(i, j));
            }
        }
    }

    SECTION("transposition") {
        M.transpose();
        int k = 1;
        for (int j = 0; j < M._n; ++j) {
            for (int i = 0; i < M._n; ++i) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i, j) == k);
                ++k;
            }
        }

        M.transpose();
        k = 1;
        for (int i = 0; i < M._n; ++i) {
            for (int j = 0; j < M._n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i, j) == k);
                ++k;
            }
        }
    }
    
    SECTION("scalar multiplication") {
        M.scale(2);

        int k = 1;
        for (int i = 0; i < M._n; ++i) {
            for (int j = 0; j < M._n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i, j) == 2*k);
                ++k;
            }
        }
    }

    SECTION("matrix addition") {
        double diag_r[5] = { 2, 2, 2, 2, 2 };
        double lower_r[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        double upper_r[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

        TriMatrix rhs(5);
        double* D_r = rhs.diag();
        for (int i = 0; i < M._n; ++i) {
            D_r[i] = diag_r[i];
        }
        double* L_r = rhs.lower();
        for (int i = 0; i < M._t; ++i) {
            L_r[i] = lower_r[i];
        }
        double* R_r = rhs.upper();
        for (int i = 0; i < M._t; ++i) {
            R_r[i] = upper_r[i];
        }

        M += rhs;
        int k = 1;
        for (int i = 0; i < M._n; ++i) {
            for (int j = 0; j < M._n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                if (i == j) {
                    CHECK(M(i, j) == k + 2);
                } else if (i > j) {
                    CHECK(M(i, j) == k + 1);
                } else {
                    CHECK(M(i, j) == k);
                }
                ++k;
            }
        }
    }

    SECTION("symmetrize matrix") {
        M.symmetrize();

        for (int i = 0; i < M._n; ++i) {
            for (int j = 0; j < M._n; ++j) {
                CAPTURE(i);
                CAPTURE(j);
                CHECK(M(i, j) == M(j, i));
            }
        }
    }

    //SECTION("symmetrize matrix, reuse storage") {
    //    M.symmetrize();

    //    CHECK(M.lower() == M.upper());
    //}
}
