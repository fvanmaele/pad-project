#include <chrono>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "lyra/lyra.hpp"
#include "malloc.h"
#include "omp.h"
#include "stencil-parallel.h"

typedef struct {
	int x;
	int y;
	int z;
	int xtile;
	int ytile;
	int ztile;
} benchParam;

void printCSVHeader() {
	std::cout << "X,Y,Z,Time[s],Bandwidth[GB/s],XTILE,YTILE,ZTILE"
		  << std::endl;
}

void printCSV(const int x, const int y, const int z, const int xtile,
              const int ytile, const int ztile, const double time,
              const double bandwidth) {
    std::cout << x << "," << y << "," << z << "," << time << ","
		  << bandwidth << "," << xtile << "," << ytile << "," << ztile
		  << std::endl;
}

void initData(const int Nx, const int Ny, const int Nz, const int radius, 
              float* A, float* B, float* vsq) {
	int offset = 0;
	
    for (int z = 0; z < Nz; ++z)
		for (int y = 0; y < Ny; ++y)
			for (int x = 0; x < Nx; ++x, ++offset) {
				if(x >= radius && x < Nx - radius &&
						y >= radius && y < Ny - radius &&
						z >= radius && z < Nz - radius){
					A[offset] = (x < Nx / 2) ? x / float(Nx)
						: y / float(Ny);
					B[offset] = 0;
					vsq[offset] = x * y * z / float(Nx * Ny * Nz);
				}
			}
}

void domainTile(std::vector<benchParam>* benchmark, const int x, const int y, const int z) {
	//Adjust this to shrink the block space 
	int xstart = 2;
	int xlim = x/2;
	int ystart = 2;
	int ylim = y/2;
	int zstart = z;
	int zlim = z;
	benchParam param;
	
    for (auto xtile = xstart; xtile <= xlim; xtile *= 2)
		for (auto ytile = ystart; ytile <= ylim; ytile *= 2)
			for (auto ztile = zstart; ztile <= zlim; ztile *= 2) {
				param.x = x;
				param.y = y;
				param.z = z;
				param.xtile = xtile;
				param.ytile = ytile;
				param.ztile = ztile;
				benchmark->push_back(param);
			}
}

void generateBenchmark(std::vector<benchParam>* benchmark, const int min,
		       const int max) {
	int x = min;
	int y = min;
	int z = min;
	while (x < max) {
		domainTile(benchmark, x, y, z);
		x *= 2;
		domainTile(benchmark, x, y, z);
		y *= 2;
		domainTile(benchmark, x, y, z);
		z *= 2;
	}
	domainTile(benchmark, max, max, max);
}

int main(int argc, char** argv) {
	using Clock = std::chrono::high_resolution_clock;
	using Duration = std::chrono::duration<double>;
	
    // Cmdline arguments
	int min = 32;
	int max = 512;
	int radius = 4;
	int steps = 1;
	int iterations = 1;
	int threads = omp_get_num_threads();
	bool show_help = false;

	/* Install lyra using vcpkg: vcpkg install lyra */

	auto cli = lyra::help(show_help) |
		   lyra::opt(min, "min")["-s"]["--min"](
		       "Start value for domain generation, default is 32") |
		   lyra::opt(max, "max")["-e"]["--max"](
		       "End value for domain generation, default is 512") |
		   lyra::opt(threads, "threads")["-n"]["--threads"](
		       "Number of threads, default is omp_get_num_threads()") |
		   lyra::opt(radius, "radius")["-r"]["--radius"](
		       "Stencil radius, default 4") |
		   lyra::opt(steps, "steps")["-t"]["--steps"](
		       "Number of time steps, default 5") |
		   lyra::opt(iterations, "iterations")["-i"]["--iterations"](
		       "Number of iterations, default is 10");

	auto result = cli.parse({argc, argv});
	if (!result) {
		std::cerr << "Error in command line: " << result.errorMessage()
			  << std::endl;
		exit(1);
	}
	if (show_help) {
		std::cout << cli << std::endl;
		exit(0);
	}

	omp_set_num_threads(threads);
	std::vector<benchParam> benchmark;
	generateBenchmark(&benchmark, min, max);
	printCSVHeader();
	
    for (auto& state : benchmark) {
		int outerX = state.x + 2*radius;
		int outerY = state.y + 2*radius;
		int outerZ = state.z + 2*radius;
		auto size = outerX * outerY * outerZ;
		
        std::vector<float> Veven(size);
		std::vector<float> Vodd(size);
		std::vector<float> Vsq(size);
		std::vector<float> coeff(radius+1);
		double time = 0;
		
        //init coefficients	
		for(auto &c : coeff)
			c = 0.1f;

		initData(outerX, outerY, outerZ, radius,Veven.data(), Vodd.data(), Vsq.data());
		for (int iter = 0; iter < iterations; ++iter) {
			auto t = Clock::now();
			loop_stencil_parallel(
					0, steps, radius, state.x+radius, radius,
					state.y+radius, radius, state.z+radius, outerX,
					outerY, outerZ, coeff.data(), Vsq.data(), Veven.data(),
					Vodd.data(), state.xtile, state.ytile, state.ztile,radius);
			Duration d = Clock::now() - t;
			time += d.count();
		}
		time /= iterations;
		double bw = (state.x * state.y * state.z * sizeof(float) * steps * 1e-9) / time;
		printCSV(state.x, state.y, state.z, state.xtile, state.ytile, state.ztile, time, bw);
	}
	return 0;
}
