#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>
#include <random>

#include "matrix/offsets.h"

using namespace asc::pad_ws20::project;

TEST_CASE("verify triangle-based symmetrization to row-major symmetrization") {
    using index_t = std::ptrdiff_t;
    auto n = GENERATE(
        1 << 5, 1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14
        );
    auto triangle_n = n*(n - 1) / 2;
    auto elements_n = n*n;

    std::vector<float> diag(n);
    std::vector<float> lower(triangle_n);
    std::vector<float> upper(triangle_n);

    // Initialize triangles with random values
    std::mt19937_64 rgen(42);
    for (index_t i = 0; i < triangle_n; ++i) {
        lower[i] = 0.5 + rgen() % 100;
        upper[i] = 1.0 + rgen() % 100;
    }
    for (index_t i = 0; i < n; ++i) {
        diag[i] = i + 1;
    }

    // For comparison to a more classic approach using row-major storage, we convert the 
    // triangle arrays to a single contiguous arrays with the proper offsets.
    std::vector<float> elements(elements_n);
    for (index_t k = 0; k < elements_n; ++k) {
        index_t j = k % n;
        index_t i = (k - j) / n;

        if (i == j) {
            elements[k] = diag[i];
        } else if (j < i) {
            elements[k] = lower[detail::offset_lower_col_major(i, j, n)];
        } else {
            elements[k] = upper[detail::offset_upper_row_major(i, j, n)];
        }
    }

    // A more typical in-place symmetrization for a row-major based matrix, iterating
    // over the lower triangle and assigning values in both triangles.
    std::vector<float> elements_sym(elements);
    for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < i; ++j) {
            index_t ij = n*i + j;
            index_t ji = n*j + i;

            float tmp = (elements_sym[ij] + elements_sym[ji]) / 2;
            elements_sym[ij] = tmp;
            elements_sym[ji] = tmp;
        }
    }

    // Verify row-major transposition
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            auto ij = n*i + j;
            auto ji = n*j + i;
            CAPTURE(ij);
            CAPTURE(ji);
            CHECK(elements_sym[ij] == Approx(elements_sym[ji]));
            CHECK(elements_sym[ij] == Approx((elements[ij] + elements[ji]) / 2));
        }
    }

    // Because lower and upper triangle and stored symmetricaly, we can symmetrize
    // the matrix as a SAXPY operation (over the lower and upper triangle) in a
    // single for loop.
    for (index_t i = 0; i < triangle_n; ++i) {
        double s = (lower[i] + upper[i]) / 2;
        lower[i] = s;
        upper[i] = s;
    }

    // Convert back to row-major array.
    for (index_t k = 0; k < elements_n; ++k) {
        index_t j = k % n;
        index_t i = (k - j) / n;

        if (i == j) {
            elements[k] = diag[i];
        } else if (j < i) {
            elements[k] = lower[detail::offset_lower_col_major(i, j, n)];
        } else {
            elements[k] = upper[detail::offset_upper_row_major(i, j, n)];
        }
    }

    // Check if converted array matches symmetrized array.
    for (index_t k = 0; k < elements_n; ++k) {
        CAPTURE(k);
        CHECK(elements[k] == elements_sym[k]);
    }
}