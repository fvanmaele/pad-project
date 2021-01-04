#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>

#include "trimatrix.h"

TEST_CASE("TriMatrix") {
    REQUIRE_THROWS(TriMatrix(-1));
    REQUIRE_THROWS(TriMatrix(0));
    REQUIRE_NOTHROW(TriMatrix(1));

    TriMatrix M(5);
    CHECK(M._n == 5);
    CHECK(M._t == 10);
    CHECK(M._s() == 25);

    double diag[5] = { 1, 7, 13, 19, 25 };
    double lower[10] = { 6, 11, 12, 16, 17, 18, 21, 22, 23, 24 };
    double upper[10] = { 2, 3, 4, 5, 8, 9, 10, 14, 15, 20 };

    double* D = M.diag();
    for (int i = 0; i < M._n; ++i)
        D[i] = diag[i];

    double* L = M.lower();
    for (int i = 0; i < M._t; ++i)
        L[i] = lower[i];

    double* R = M.upper();
    for (int i = 0; i < M._t; ++i)
        R[i] = upper[i];

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

    }
}
