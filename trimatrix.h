#ifndef TRIMATRIX_H
#define TRIMATRIX_H

#include <memory>

class TriMatrix
{
public:
    typedef std::unique_ptr<double> ptr;
    TriMatrix(ptrdiff_t n);
    double operator()(ptrdiff_t i, ptrdiff_t j) const;
    double& operator()(ptrdiff_t i, ptrdiff_t j);

    void scale(double a);
    TriMatrix& operator+=(const TriMatrix& rhs);
    void transpose() { 
        _lower.swap(_upper);
    }
    void symmetrize();

    double* diag() noexcept { 
        return _diag.get();
    }
    const double* diag() const noexcept {
        return _diag.get();
    }   
    double* lower() noexcept { 
        return _lower.get();
    }
    const double* lower() const noexcept {
        return _lower.get();
    }   
    double* upper() noexcept { 
        return _upper.get();
    }
    const double* upper() const noexcept { 
        return _upper.get();
    }

    const ptrdiff_t _n;
    const ptrdiff_t _t;

    ptrdiff_t _s() const noexcept { 
        return _n + 2*_t;
    }

private:
    std::unique_ptr<double> _diag;
    std::unique_ptr<double> _lower;
    std::unique_ptr<double> _upper;
};

#endif // TRIMATRIX_H
