#include <iostream>
#include <random>
#include <cassert>
#include <utility>

#include <upcxx/upcxx.hpp>

int main(int argc, char** argv) 
{
    // BEGIN PARALLEL REGION
    upcxx::init();
    const long N = 2 << 10;
    const int seed = 42;

    // Block size for each process
    const long block_size = N / upcxx::rank_n();
    assert(block_size % 2 == 0);
    assert(N == block_size * upcxx::rank_n());

    // Allocate array, divided between processes
    upcxx::global_ptr<float> u_g(upcxx::new_array<float>(block_size));

    // Initialize blocks with random values (per-process)
    assert(u_g.is_local()); // ensure global pointer has affinity to a local process
    float* u = u_g.local(); // downcast to local pointer
    
    std::mt19937_64 rgen(seed);
    for (long i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }

    // Compute partial sums, per-process. Use a distributed object for later reduction
    // of partial sums. Downcasting to a raw pointer is possible, because we only use
    // the block of the array local to the process for summation.
    upcxx::dist_object<upcxx::global_ptr<double> > psum_dist(upcxx::new_<double>(0));
    assert(psum_dist->is_local());
    double* psum = psum_dist->local();

    for (long i = 0; i < block_size; ++i) {
        *psum += u[i];
    }
    upcxx::barrier(); // ensure all partial sums are available
    std::cout << *psum << " (Rank " << upcxx::rank_me() << ")" << std::endl;

    // Reduce partial sums through dist_object::fetch (communication) on master process
    if (upcxx::rank_me() == 0) {
        // partial sum for process 0
        double res = *psum;
        // partial sums for remaining processes
        for (int k = 1; k < upcxx::rank_n(); ++k) {
            // get pointer to remote psum
            upcxx::global_ptr<double> psum_ptr_k = psum_dist.fetch(k).wait();
            // get value of remote psum
            double psum_k = upcxx::rget(psum_ptr_k).wait();
            // finally, add value to result
            res += psum_k;
        }
        std::cout << res << std::endl;
    }

    upcxx::delete_array(u_g);

    upcxx::finalize();
    // END PARALLEL REGION
}