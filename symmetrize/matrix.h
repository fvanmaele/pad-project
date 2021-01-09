#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <memory>
#include <limits>
#include <gsl/gsl-lite.hpp>
#include <utility> // for swap

namespace PAD
{
class SquareMatrix
{
public:
    SquareMatrix(ptrdiff_t n)  // elements in row-major order
        : _n(n), _s(n*n)
    {
        gsl_Expects(_n >= 1);
        gsl_Expects(_n < std::numeric_limits<ptrdiff_t>::max() / _n);

        _elements = std::unique_ptr<double>(new double[_s]);
    }

    double operator()(ptrdiff_t i, ptrdiff_t j) const {
        gsl_Expects(i >= 0 && i < _n);
        gsl_Expects(j >= 0 && j < _n);

        return _elements.get()[_n*i + j];
    }

    double& operator()(ptrdiff_t i, ptrdiff_t j) {
        gsl_Expects(i >= 0 && i < _n);
        gsl_Expects(j >= 0 && j < _n);

        return _elements.get()[_n*i + j];
    }

    void transpose() {
        double* elems = _elements.get();

        for (ptrdiff_t i = 0; i < _n ; ++i) {
            // Iterate across lower triangle
            for (ptrdiff_t j = 0; j < i; ++j) {
                ptrdiff_t ij = _n*i + j;
                ptrdiff_t ji = _n*j + i;

                using std::swap;
                swap(elems[ij], elems[ji]);
            }
        }
    }

    void symmetrize() {
        double* elems = _elements.get();

        for (ptrdiff_t i = 0; i < _n; ++i) {
            // Iterate across lower triangle
            for (ptrdiff_t j = 0; j < i; ++j) {
                ptrdiff_t ij = _n*i + j;
                ptrdiff_t ji = _n*j + i;

                double tmp = (elems[ij] + elems[ji]) / 2;
                elems[ij] = tmp;
                elems[ji] = tmp;
            }
        }
    }

    ptrdiff_t n() const noexcept { return _n; }
    ptrdiff_t t() const noexcept { return _n*(_n - 1) / 2; }
    ptrdiff_t s() const noexcept { return _s; }
    double* elements() { return _elements.get(); }

private:
    ptrdiff_t _n;
    ptrdiff_t _s;
    std::unique_ptr<double> _elements; // row-major order
};

}
#endif // MATRIX_H
