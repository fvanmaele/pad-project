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

double TriMatrix::operator()(ptrdiff_t i, ptrdiff_t j) const
{
    gsl_Expects(i >= 0 && i < _n);
    gsl_Expects(j >= 0 && j < _n);

    if (i == j) {
        return _diag.get()[i];
    } else if (i > j) { // row-major order
        return _lower.get()[offset_lower_row_major(i, j)];
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
        return _lower.get()[offset_lower_row_major(i, j)];
    } else { // j > i
        return _upper.get()[offset_upper_row_major(i, j, _n)];
    }
}

void TriMatrix::scale(double a)
{
    for (ptrdiff_t i = 0; i < _n; ++i)
        _diag.get()[i] *= a;
    for (ptrdiff_t i = 0; i < _t; ++i)
        _lower.get()[i] *= a;
    for (ptrdiff_t i = 0; i < _t; ++i)
        _upper.get()[i] *= a;
}

TriMatrix &TriMatrix::operator+=(const TriMatrix &rhs)
{
    const double* rhs_diag = rhs.diag();
    const double* rhs_lower = rhs.lower();
    const double* rhs_upper = rhs.upper();

    for (ptrdiff_t i = 0; i < _n; ++i)
        _diag.get()[i] += rhs_diag[i];
    for (ptrdiff_t i = 0; i < _t; ++i)
        _lower.get()[i] += rhs_lower[i];
    for (ptrdiff_t i = 0; i < _t; ++i)
        _upper.get()[i] += rhs_upper[i];
    return *this;
}

void TriMatrix::symmetrize()
{

}
