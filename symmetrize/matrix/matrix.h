#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <cassert>
#include <memory>
#include <limits>
#include <type_traits>
#include <iterator>
#include <iostream>
#include <utility>


namespace asc::pad_ws20::project
{
template <typename T>
class SquareMatrix
{
public:
    static_assert(std::is_arithmetic_v<T>);
    typedef T value_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::ptrdiff_t index_t;

    SquareMatrix(index_t n)
        : _n(n), _s(n*n)
    {
        assert(_n >= 1);
        assert(_n < std::numeric_limits<index_t>::max() / _n); // overflow check
        _elements = std::unique_ptr<T>(new T[_s]);
    }

    SquareMatrix(T* row_major, index_t length) {
        assert(length >= 1);
        _n = static_cast<index_t>(std::sqrt(length));

        assert(length == _n*_n);
        _s = length;
        _elements = std::unique_ptr<T>(new T[_s]);

        T* _e = _elements.get();
        for (index_t k = 0; k < length; ++k) {
            _e[k] = row_major[k];
        }
    }

    SquareMatrix(const SquareMatrix&) = delete;
    SquareMatrix& operator=(const SquareMatrix&) = delete;
    SquareMatrix(SquareMatrix&&) = default;
    SquareMatrix& operator=(SquareMatrix&&) = default;
    ~SquareMatrix() = default;

    T operator()(index_t i, index_t j) const {
        assert(i >= 0 && i < _n);
        assert(j >= 0 && j < _n);

        return _elements.get()[_n*i + j];
    }

    T& operator()(index_t i, index_t j) {
        assert(i >= 0 && i < _n);
        assert(j >= 0 && j < _n);

        return _elements.get()[_n*i + j];
    }

    void transpose() {
        T* elems = _elements.get();

        for (index_t i = 0; i < _n ; ++i) {
            // Iterate across lower triangle
            for (index_t j = 0; j < i; ++j) {
                index_t ij = _n*i + j;
                index_t ji = _n*j + i;

                using std::swap;
                swap(elems[ij], elems[ji]);
            }
        }
    }

    void symmetrize() {
        T* elems = _elements.get();

        for (index_t i = 0; i < _n; ++i) {
            // Iterate across lower triangle
            for (index_t j = 0; j < i; ++j) {
                index_t ij = _n*i + j;
                index_t ji = _n*j + i;

                T tmp = (elems[ij] + elems[ji]) / 2;
                elems[ij] = tmp;
                elems[ji] = tmp;
            }
        }
    }

    index_t n() const noexcept { return _n; }
    index_t t() const noexcept { return _n*(_n - 1) / 2; }
    index_t s() const noexcept { return _s; }
    T* elements() { return _elements.get(); }

private:
    index_t _n;
    index_t _s;
    std::unique_ptr<T> _elements; // row-major order
};

} // namespace asc::pad_ws20::upcxx

#endif // MATRIX_H
