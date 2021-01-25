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

// Reduction per process (local array)
int main(int argc, char** argv) {
    // BEGIN PARALLEL REGION
    upcxx::init();
    const long N = 2 << 10;
    const long block_size = N / upcxx::rank_n();

    assert(block_size % 2 == 0);
    assert(N == block_size * upcxx::rank_n());

    // Initialize array with random values
    upcxx::global_ptr<float> u_g(upcxx::new_array<float>(block_size));
    float* u = smp_init_random(u_g, block_size, upcxx::rank_me());

    // Compute partial sums
    double psum(0);

    for (long i = 0; i < block_size; ++i) {
        psum += u[i];
    }
    upcxx::barrier(); // ensure all partial sums are available

    std::cout << psum << " (Rank " << upcxx::rank_me() << ")" << std::endl;

    // Initialize reduction value on main thread
    upcxx::global_ptr<double> res_g = nullptr;
    if (upcxx::rank_me() == 0) {
        res_g = upcxx::new_<double>(0);
    }

    // Broadcast reduction value to all processes
    res_g = upcxx::broadcast(res_g, 0).wait();
    double* res = res_g.local();

    // Add partial sums process-by-process (avoid concurrent writes)
    for (int k = 0; k < upcxx::rank_n(); ++k) {
        if (upcxx::rank_me() == k) {
            *res += psum;
        }
        upcxx::barrier();
    }
    std::cout << *res << std::endl;

    upcxx::delete_array(u_g);

    upcxx::finalize();
    // END PARALLEL REGION
}
