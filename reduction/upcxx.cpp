#include <iostream>
#include <random>
#include <cassert>
#include <utility>

#include <upcxx/upcxx.hpp>

#include "accumulate.h"

// Compute partial sums for process
double smp_reduce(long N, long block_size, float* u) 
{
    double res_process(0);

    for (long i = 0; i < block_size; ++i) {
        res_process += u[i];
    }
    return res_process;
}

// Compute partial sums for process
double smp_reduce_shared(long N, long block_size, float* u_node)
{
    double res_process(0);
    long offset = upcxx::rank_me() * block_size;

    for (long i = 0; i < block_size; ++i) {
        res_process += u_node[offset + i];
    }
    return res_process;
}

double dist_reduce(long N, long block_size, float* u)
{
    return {}; // TODO
}

double dist_reduce_shared(long node_size, long block_size, float* u_node)
{
    return {}; // TODO
}

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

void smp_bandwidth(long N)
{
    // TODO
}

void smp_bandwidth_shared(long N)
{
    // TODO
}

void dist_bandwidth(long N)
{
    // TODO
}

void builtin_bandwidth(long N)
{
    // TODO
}

int main() {
    // XXX: compare result against serial implementation (for lower N?)

    // Begin parallel region
    upcxx::init();
    const long N = 2 << 10;
    const long block_size = N / upcxx::rank_n();

    assert(block_size % 2 == 0);
    assert(N == block_size * upcxx::rank_n());

    // Reduction per process (local array)
    {
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
        // XXX: some way to do this with upcxx::dist_object?
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

        upcxx::delete_array(u_g);
    }

    // Reduction per process (shared array)
    {
        int g_id = upcxx::rank_me();
        int g_cnt = upcxx::rank_n();
        
        upcxx::global_ptr<float> u_g;
        float* u;
        
        // Initialize shared array in master process
        if (g_id == 0) {
            u_g = upcxx::new_array<float>(N);
            u = smp_init_random(u_g, N, 0);
        }

        // Broadcast shared array to other processes
        u_g = upcxx::broadcast(u_g, 0).wait();
        u = u_g.local();

        // Compute partial sums
        double psum = smp_reduce_shared(N, block_size, u);
        upcxx::barrier(); // ensure all partial sums are available

        std::cout << psum << std::endl;

        // Initialize reduction value on main thread
        upcxx::global_ptr<double> res_g = nullptr;
        if (upcxx::rank_me() == 0) {
            res_g = upcxx::new_<double>(0);
        }

        // Broadcast reduction value to all processes
        // XXX: some way to do this with upcxx::dist_object?
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
    }
    
    // Reduction per node (avoid communication overhead for RMA on local shared memory)
    // {
    //     int l_id = upcxx::local_team().rank_me();
    //     int l_cnt = upcxx::local_team().rank_n();
    //     // XXX: assumes homogenous distribution of processes between nodes
    //     const long node_size = N / (upcxx::rank_n() / l_cnt);
    //     assert(node_size %2 == 0);
    //     assert(N == node_size * (upcxx::rank_n() / l_cnt));
        
    //     upcxx::global_ptr<float> gp_data;
    //     if (l_id == 0) {
    //         gp_data = upcxx::new_array<float>(node_size);
    //         // Chooses a different seed for each node, because the node id is not available
    //         // and can not be passed to std::mersenne_twister_engine::discard
    //         // FIXME: Not deterministic and so cannot be used to check results with serial case (fixed seed)
    //         smp_init_random(gp_data, node_size, 0, upcxx::rank_me());
    //     }
    //     gp_data = upcxx::broadcast(gp_data, 0, upcxx::local_team()).wait();
    // }

    upcxx::finalize();
}
// XXX: add serial test
// int main() {
//     upcxx::init();
//     // Initialize parameters - simple test case
//     const long N = 2 << 10;
//     // XXX: case that rank_n does not divide n
//     const long N_local = N / upcxx::rank_n();
//     assert(N_local % 2 == 0);
//     assert(N == N_local * upcxx::rank_n());

//     // Allocate and share one array (per node)
//     int l_id = upcxx::rank_me(); // upcxx::local_team().rank_me();
//     int l_cnt = upcxx::rank_n(); // upcxx::local_team().rank_n();
//     // upcxx::global_ptr<float> gp_data;
//     // if (l_id == 0) {
//     //     gp_data = new_array<float>(N_local);
//     // }
//     // gp_data = upcxx::broadcast(gp_data, 0).wait(); // broadcast(gp_data, 0, upcxx::local_team()).wait();

//     // Allocate one array per process
//     upcxx::global_ptr<float> u_g(upcxx::new_array<float>(N_local));
//     float* u = u_g.local();
//     double res_process(0); // partial sum

//     // Init to uniform pseudo-random distribution, independent of job size
//     // std::mt19937_64 rgen(1);
//     // rgen.discard(upcxx::rank_me() * N_local);

//     for (long i = 0; i < N_local; ++i) {
//         //u[i] = 0.5 + rgen() % 100;
//         u[i] = 1;
//     }

//     // Value that holds final result (master process)
//     upcxx::global_ptr<double> res_g;
//     if (l_id == 0) {
//         res_g = upcxx::new_<double>(0);
//     }
//     res_g = upcxx::broadcast(res_g, 0).wait();
//     double* res = res_g.local();
    
//     // Compute partial sums and add to total
//     for (long i = 0; i < N_local; ++i) {
//         res_process += u[i];
//     }
//     *res += res_process; // direct store to value created by node leader
    
//     upcxx::barrier(); // ensure all additions are completed

//     if (l_id == 0)
//         std::cout << *res << std::endl;
//     // Add partial sums
    
//     upcxx::finalize();
// }