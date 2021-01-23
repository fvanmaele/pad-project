#include <iostream>
#include <random>
#include <cassert>
#include <utility>

#include <upcxx/upcxx.hpp>

#include "accumulate.h"

// Initialize blocks with random values
float* smp_init_random(upcxx::global_ptr<float> u_g, long block_size, 
                       upcxx::intrank_t rank = 0, size_t seed = 42) 
{
    assert(u_g.is_local()); // ensure global pointer has affinity to a local process
    float* u = u_g.local(); // downcast to local pointer

    std::mt19937_64 rgen(seed);
    rgen.discard(rank * block_size); // rank = 0 -> do not discard

    for (long i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }
    return u;
}

int main(int argc, char** argv) 
{

}