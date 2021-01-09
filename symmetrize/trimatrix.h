#ifndef TRIMATRIX_H
#define TRIMATRIX_H

#include <cstddef>
#include <memory>
#include <gsl/gsl-lite.hpp>
#include <limits>

namespace PAD 
{

inline ptrdiff_t offset_lower_col_major(ptrdiff_t i, ptrdiff_t j, ptrdiff_t n) {
    // 1st summand: expansion of n*(n-1)/2 - (n-1-j)*(n-j)/2
    // 2nd summand: offset for i
    return j*(2*n - 1 - j)/2 + (i - j - 1);
}

inline ptrdiff_t offset_upper_row_major(ptrdiff_t i, ptrdiff_t j, ptrdiff_t n) {
    // 1st summand: expansion of n*(n-1)/2 - (n-1-i)*(n-i)/2
    // 2nd summand: offset for j
    return i*(2*n - 1 - i)/2 + (j - i - 1);
}

struct TriMatrix
{
    TriMatrix(ptrdiff_t n) 
        : _n(n), _t(n*(n-1) / 2)
    {
        gsl_Expects(n >= 1);
        _diag  = std::unique_ptr<double>(new double[_n]);

        if (n >= 2) {
            gsl_Expects(_n < std::numeric_limits<ptrdiff_t>::max() / (_n - 1));
            _lower = std::unique_ptr<double>(new double[_t]);
            _upper = std::unique_ptr<double>(new double[_t]);
        }
    }
 
    double operator()(ptrdiff_t i, ptrdiff_t j) const
    {
        gsl_Expects(i >= 0 && i < _n);
        gsl_Expects(j >= 0 && j < _n);

        if (i == j) {
            return _diag.get()[i];
        } else if (i > j) { // row-major order
            return _lower.get()[offset_lower_col_major(i, j, _n)];
        } else { // j > i
            return _upper.get()[offset_upper_row_major(i, j, _n)];
        }
    }

    double& operator()(ptrdiff_t i, ptrdiff_t j)
    {
        gsl_Expects(i >= 0 && i < _n);
        gsl_Expects(j >= 0 && j < _n);

        if (i == j) {
            return _diag.get()[i];
        } else if (i > j) { // row-major order
            return _lower.get()[offset_lower_col_major(i, j, _n)];
        } else { // j > i
            return _upper.get()[offset_upper_row_major(i, j, _n)];
        }
    }

    void transpose() {
        _lower.swap(_upper);
    }

    void symmetrize() {
        double* _l = _lower.get();
        double* _u = _upper.get();

        for (ptrdiff_t i = 0; i < _t; ++i) {
            double s = (_l[i] + _u[i]) / 2.;
            _l[i] = s;
            _u[i] = s;
        }
    }

    double* diag()  { return _diag.get(); }
    double* lower() { return _lower.get(); }
    double* upper() { return _upper.get(); }

    ptrdiff_t n() const noexcept { return _n; }
    ptrdiff_t t() const noexcept { return _t; }
    ptrdiff_t s() const noexcept { return _n + 2*_t; }

private:
    ptrdiff_t _n;
    ptrdiff_t _t;
    std::unique_ptr<double> _diag;
    std::unique_ptr<double> _lower;
    std::unique_ptr<double> _upper;
};

} // namespace PAD

#endif // TRIMATRIX_H
