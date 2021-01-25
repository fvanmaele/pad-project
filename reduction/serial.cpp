
#include <vector>
#include <random>
#include <iostream>

int main(int argc, char** argv) {
    const size_t seed = 42;
    std::mt19937_64 rgen(seed);
    
    const long N = 2 << 10;
    std::vector<float> u(N);

    //rgen.discard(rank * block_size);
    
    // XXX: use normal distribution
    for (auto&& c: u) {
        c = 0.5 + rgen() % 100;
    }
    
    double sum = 0;
    for (auto&& c: u) {
        sum += c;
    }
    std::cout << sum << std::endl;
}