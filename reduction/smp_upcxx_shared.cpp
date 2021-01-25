
#include <iostream>
#include <random>
#include <cassert>
#include <utility>

#include <upcxx/upcxx.hpp>

// Initialize blocks with random values
void smp_init_random(float* u, long block_size, upcxx::intrank_t rank = 0, size_t seed = 42) 
{
    std::mt19937_64 rgen(seed);
    rgen.discard(rank * block_size); // rank = 0 -> do not discard

    for (long i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }
}

// Reduction per process (shared array)
int main(int argc, char** argv) {
    // BEGIN PARALLEL REGION
    upcxx::init();
    int proc_id = upcxx::rank_me();
    int nproc = upcxx::rank_n();
    
    const long N = 2 << 18;
    const long block_size = N / nproc;

    assert(block_size % 2 == 0);
    assert(N == block_size * nproc);
    
    // Initialize shared array in process 0
    upcxx::global_ptr<float> u_g;
    float* u;

    if (proc_id == 0) {
        u_g = upcxx::new_array<float>(N);
        assert(u_g.is_local()); // ensure global pointer has affinity to a local process
        u = u_g.local();        // downcast to local pointer

        smp_init_random(u, N, 0);
    }

    // Broadcast shared array to other processes
    u_g = upcxx::broadcast(u_g, 0).wait();
    assert(u_g.is_local());
    u = u_g.local();

    // Create a reduction value for each process (universal name, local value).
    // Each local value can be accessed with operator* or operator->, but there
    // is no guarantee that every local value is constructed after the call.
    upcxx::dist_object<double> psum_d(0);
    upcxx::barrier();
    
    long offset = upcxx::rank_me() * block_size;
    for (long i = 0; i < block_size; ++i) {
        *psum_d += u[offset + i];
    }
    upcxx::barrier(); // ensure all partial sums are available
    std::cout << *psum_d << std::endl;
    
    if (proc_id == 0) {
        // partial sum for process 0
        double res(*psum_d);

        // partial sum for other processes
        for (int k = 1; k < upcxx::rank_n(); ++k) {
            double psum = psum_d.fetch(k).wait();
            res += psum;
        }
        std::cout << res << std::endl;
    }

    if (proc_id == 0) {
        upcxx::delete_array(u_g);
    }

    upcxx::finalize();
    // END PARALLEL REGION
}
