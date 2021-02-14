#ifndef UPCXX_PRINT_HPP
#define UPCXX_PRINT_HPP
#include <iostream>
#include <fstream>
#include "upcxx.hpp"

inline std::ostream& 
dump_array(std::ostream &stream, float* array, index_t begin, index_t end)
{
    if (stream) {
        for (index_t i = begin; i != end-1; ++i) {
            stream << array[i] << " ";
        }
        stream << array[end-1];
    }
    return stream;
}

inline void
dump_stencil_impl(std::ostream &stream, float* array, index_t n_local, index_t n_ghost_offset, const char *label)
{
    const upcxx::intrank_t proc_id = upcxx::rank_me();
    const upcxx::intrank_t proc_n = upcxx::rank_n();

    for (int k = 0; k < proc_n; ++k) {
        if (proc_id == k) {
            if (k == 0) {
                stream << label;
                dump_array(stream, array, 0, n_local - n_ghost_offset);
                stream << std::flush; // avoid mangling output
            } else if (k == proc_n - 1) {
                stream << " ";
                dump_array(stream, array, n_ghost_offset, n_local);
                stream << std::endl;
            } else {
                stream << " ";
                dump_array(stream, array, n_ghost_offset, n_local - n_ghost_offset);
                stream << std::flush;
            }
        }
        upcxx::barrier();
    }
}

inline void
dump_stencil(float* Veven, float* Vodd, float* Vsq, index_t n_local, index_t n_ghost_offset, const char* file_path)
{
    if (upcxx::rank_me() == 0) {
        std::ofstream ofs(file_path, std::ofstream::trunc);
    }
    upcxx::barrier();
    std::ofstream ofs(file_path, std::ofstream::app);

    dump_stencil_impl(ofs, Veven, n_local, n_ghost_offset, "Veven: ");
    dump_stencil_impl(ofs, Vodd, n_local, n_ghost_offset, "Vodd: ");
    dump_stencil_impl(ofs, Vsq, n_local, n_ghost_offset, "Vsq: ");
}

#endif // UPCXX_PRINT_HPP