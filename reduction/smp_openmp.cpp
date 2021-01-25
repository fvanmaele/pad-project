
#include <random>
#include <iostream>

int main(int argc, char** argv) {
    const size_t seed = 42;
    std::mt19937_64 rgen(seed);
    
    const long N = 2 << 10;
    //const long block_size = N / upcxx::rank_n();
    float* u = new float[N];
    //rgen.discard(rank * block_size);
    
    // XXX: use normal distribution
    for (long i = 0; i < N; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }
    
    double sum = 0;
    #pragma omp parallel for simd reduction(+: sum)
    for (long i = 0; i < N; ++i) {
        sum += u[i];
    }
    std::cout << sum << std::endl;
}