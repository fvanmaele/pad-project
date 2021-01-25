
#include <iostream>
#include <random>
#include <cassert>
#include <utility>

#include <upcxx/upcxx.hpp>

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

// Reduction per process (shared array)
int main(int argc, char** argv) {
    // BEGIN PARALLEL REGION
    upcxx::init();
    const long N = 2 << 10;
    const long block_size = N / upcxx::rank_n();

    assert(block_size % 2 == 0);
    assert(N == block_size * upcxx::rank_n());

    int g_id = upcxx::rank_me();
    int g_cnt = upcxx::rank_n();
    
    // Initialize shared array in master process
    upcxx::global_ptr<float> u_g;
    float* u;

    if (g_id == 0) {
        u_g = upcxx::new_array<float>(N);
        u = smp_init_random(u_g, N, 0);
    }

    // Broadcast shared array to other processes
    u_g = upcxx::broadcast(u_g, 0).wait();
    u = u_g.local();

    // Compute partial sums
    double psum = 0;
    long offset = upcxx::rank_me() * block_size;

    for (long i = 0; i < block_size; ++i) {
        psum += u[offset + i];
    }
    upcxx::barrier(); // ensure all partial sums are available

    std::cout << psum << std::endl;

    // Initialize reduction value on main thread
    upcxx::global_ptr<double> res_g = nullptr;
    if (upcxx::rank_me() == 0) {
        res_g = upcxx::new_<double>(0);
    }

    // Broadcast reduction value to all processes
    res_g = upcxx::broadcast(res_g, 0).wait();
    double* res = res_g.local();

    // Add partial sums process-by-process
    for (int k = 0; k < upcxx::rank_n(); ++k) {
        if (upcxx::rank_me() == k) {
            *res += psum;
        }
        upcxx::barrier();
    }
    std::cout << *res << std::endl;

    if (g_id == 0) {
        upcxx::delete_array(u_g);
    }

    upcxx::finalize();
    // END PARALLEL REGION
}
