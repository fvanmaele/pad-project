#include <cstddef>
#include <limits>
#include <gsl/gsl-lite.hpp>

#include "trimatrix.h"


TriMatrix::TriMatrix(ptrdiff_t n)
    : _n(n), _t(n*(n-1) / 2)
{
    gsl_Expects(_n >= 1);
    if (_n >= 2) {
        gsl_Expects(_n < std::numeric_limits<ptrdiff_t>::max() / (_n - 1));
    }

    _diag = std::unique_ptr<double>(new double[_n]);
    _lower = std::unique_ptr<double>(new double[_t]);
    _upper = std::unique_ptr<double>(new double[_t]);
}

ptrdiff_t offset_lower_row_major(ptrdiff_t i, ptrdiff_t j) {
    // 1st summand: sum 0+1+...+(i-1)
    // 2nd summand: offset for j
    return i*(i - 1)/2 + j;
}

ptrdiff_t offset_upper_row_major(ptrdiff_t i, ptrdiff_t j, ptrdiff_t n) {
    // 1st summand: expansion of n*(n-1)/2 - (n-1-i)*(n-i)/2
    // 2nd summand: offset for j
    return i*(2*n - 1 - i)/2 + (j - i - 1);
}

ptrdiff_t offset_lower_col_major(ptrdiff_t i, ptrdiff_t j, ptrdiff_t n) {
    // 1st summand: expansion of n*(n-1)/2 - (n-1-j)*(n-j)/2
    // 2nd summand: offset for i
    return j*(2*n - 1 - j)/2 + (i - j - 1);
}

ptrdiff_t offset_upper_col_major(ptrdiff_t i, ptrdiff_t j) {
    // 1st summand: sum 0+1+...+(j-1)
    // 2nd summand: offset for i
    return j*(j - 1)/2 + i;
}

double TriMatrix::operator()(ptrdiff_t i, ptrdiff_t j) const
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

double& TriMatrix::operator()(ptrdiff_t i, ptrdiff_t j)
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

void TriMatrix::scale(double a)
{
    double* diag_ptr = _diag.get();
    for (ptrdiff_t i = 0; i < _n; ++i) {
        diag_ptr[i] *= a;
    }
    
    double* lower_ptr = _lower.get();
    for (ptrdiff_t i = 0; i < _t; ++i) {
        lower_ptr[i] *= a;
    }
    
    double* upper_ptr = _upper.get();
    for (ptrdiff_t i = 0; i < _t; ++i) {
        upper_ptr[i] *= a;
    }
}

TriMatrix &TriMatrix::operator+=(const TriMatrix &rhs)
{
    const double* rhs_diag = rhs.diag();
    double* lhs_diag = diag();
    for (ptrdiff_t i = 0; i < _n; ++i) {
        lhs_diag[i] += rhs_diag[i];
    }
    
    const double* rhs_lower = rhs.lower();
    double* lhs_lower = lower();
    for (ptrdiff_t i = 0; i < _t; ++i) {
        lhs_lower[i] += rhs_lower[i];
    }
    
    const double* rhs_upper = rhs.upper();
    double* lhs_upper = upper();
    for (ptrdiff_t i = 0; i < _t; ++i) {
        lhs_upper[i] += rhs_upper[i];
    }
    return *this;
}

void TriMatrix::symmetrize()
{
    double* diag_ptr = _diag.get();
    for (ptrdiff_t i = 0; i <_n; ++i) {
        diag_ptr[i] /= 2.;
    }

    double* _l = _lower.get();
    double* _u = _upper.get();
    for (ptrdiff_t i = 0; i <_t; ++i) {
        double s = (_l[i] + _u[i]) / 2.;
        _l[i] = s;
        _u[i] = s;
    }
}
