#include <iostream>
#include <random>
#include <cassert>
#include <utility>
#include <string>

#include <cstdlib>
#include <getopt.h>
#include <upcxx/upcxx.hpp>

int main(int argc, char** argv) 
{
    long N = 0;     // array size
    int seed = 42;  // seed for pseudo-random generator

    struct option long_options[] = {
        { "size", required_argument, NULL, 's' },
        { "seed", optional_argument, NULL, 't' },
        { NULL, 0, NULL, 0 }
    };

    int c;
    while ((c = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
        switch(c) {
            case 's':
                N = std::stol(optarg);
                break;
            case 't':
                seed = std::stoi(optarg);
                break;
            case '?':
                break;
            default:
                std::terminate();
        }
    }
    if (N <= 0) {
        throw std::invalid_argument("a positive array size is required");
    }

    // BEGIN PARALLEL REGION
    upcxx::init();
    int nproc = upcxx::rank_n();                    // amount of processes
    int nproc_node = upcxx::local_team().rank_n();  // amount of processes in each node
    int nodes = nproc / nproc_node;                 // amount of nodes
    
    int proc_id = upcxx::rank_me();
    int proc_id_node = upcxx::local_team().rank_me();

    // We assume the amount of processes is a multiple of the amount of nodes
    assert(nproc == nodes * nproc_node);

    // Block size for each node
    const long node_size = N / nodes;
    assert(node_size % 2 == 0);
    assert(N == node_size * nodes);

    // Block size for each process
    const long block_size = node_size / nproc_node;
    assert(block_size % 2 == 0);
    assert(node_size == block_size * nproc_node);

    // Initialize shared array in process 0 of each node (local_team())
    upcxx::global_ptr<float> u_node;
    if (proc_id_node == 0) {
        u_node = upcxx::new_array<float>(node_size);
    }

    // Broadcast array to other processes in the node such that they
    // can access it directly. As multiple processes share a single array,
    // care should be taken to protect against concurrent writes.
    u_node = upcxx::broadcast(u_node, 0, upcxx::local_team()).wait();
    assert(u_node.is_local());
    float* u = u_node.local();

    // Every process now initializes a seperate block of the shared array
    // (in the corresponding node) with random values
    size_t offset = proc_id_node * block_size;
    std::mt19937_64 rgen(seed);
    rgen.discard(offset);

    for (long i = 0; i < block_size; ++i) {
        u[offset + i] = 0.5 + rgen() % 100;
    }

    // Create a reduction value for each process (universal name, local value)
    upcxx::dist_object<double> psum_d(0);
    upcxx::barrier();

    for (long i = 0; i < block_size; ++i) {
        *psum_d += u[offset + i];
    }

    // Ensure all partial sums are available
    upcxx::barrier();
    std::cout << *psum_d << std::endl;

    // Reduce partial sums through dist_object::fetch (communication) on master process
    if (upcxx::rank_me() == 0) {
        // partial sum for process 0
        double res(*psum_d);

        // partial sums for remaining processes
        for (int k = 1; k < upcxx::rank_n(); ++k) {
            double psum = psum_d.fetch(k).wait();
            res += psum;
        }
        std::cout << res << std::endl;
    }
}