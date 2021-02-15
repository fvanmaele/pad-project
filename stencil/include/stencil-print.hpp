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
dump_stencil_impl(std::ostream &stream, float* array, index_t n_local, index_t n_ghost_offset, 
                 const char *label, bool print_all)
{
    const upcxx::intrank_t proc_id = upcxx::rank_me();
    const upcxx::intrank_t proc_n = upcxx::rank_n();

    if (proc_n == 1) {
        stream << label << ": ";
        dump_array(stream, array, 0, n_local);
        stream << std::endl; // avoid mangling output
        return;
    }

    if (print_all) {
        for (int k = 0; k < proc_n; ++k) {
            if (proc_id == k) {
                stream << "Rank " << proc_id << std::endl;
                stream << label << std::endl;

                stream << "Ghost (lower): ";
                dump_array(stream, array, 0, n_ghost_offset);
                stream << std::endl;
                
                stream << "Block: ";
                dump_array(stream, array, n_ghost_offset, n_local - n_ghost_offset);
                stream << std::endl;
                
                stream << "Ghost (upper): ";
                dump_array(stream, array, n_local - n_ghost_offset, n_local);
                stream << std::endl << std::endl;
            }
            upcxx::barrier();
        }
    }  else {
        for (int k = 0; k < proc_n; ++k) {
            if (proc_id == k) {
                if (k == 0) {
                    stream << label << ": ";
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
}

inline void
dump_stencil(float* Veven, float* Vodd, float* Vsq, index_t n_local, index_t n_ghost_offset, 
             const char* file_path, bool print_all = false)
{
    if (upcxx::rank_me() == 0) {
        std::ofstream ofs;
        ofs.exceptions(std::ofstream::badbit);
        ofs.open(file_path, std::ofstream::trunc);
    }
    upcxx::barrier();
    std::ofstream ofs(file_path, std::ofstream::app);
    if (ofs) {
        dump_stencil_impl(ofs, Veven, n_local, n_ghost_offset, "Veven", print_all);
        dump_stencil_impl(ofs, Vodd, n_local, n_ghost_offset, "Vodd", print_all);
        dump_stencil_impl(ofs, Vsq, n_local, n_ghost_offset, "Vsq", print_all);
    }
}

#endif // UPCXX_PRINT_HPP