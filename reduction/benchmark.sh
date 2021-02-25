#!/bin/bash
set -e
type upcxx-run
type cmake
type ninja
[[ -r $HOME/source/vcpkg/scripts/buildsystems/vcpkg.cmake ]]

# Enabled benchmarks (shared)
run_upcxx_skl=1
run_upcxx_knl=1
run_openmp_skl=1
run_openmp_knl=1

# Enabled benchmarks (distributed)
run_upcxx_skl_dist=1
run_upcxx_knl_dist=1
run_openmp_skl_dist=1
run_openmp_knl_dist=1

# Number of iterations
iterations=100

cmake() {
    command cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE="$HOME/source/vcpkg/scripts/buildsystems/vcpkg.cmake" "$@"
}

[[ -d build-dist ]] && rm -rf build-dist
mkdir build-dist
[[ -d build-shared ]] && rm -rf build-shared
mkdir build-shared


# ---------------------------------------
# SHARED MEMORY
# ---------------------------------------
cd build-shared
UPCXX_NETWORK=smp cmake -DCMAKE_BUILD_TYPE=Release ../..
ninja -v reduction-upcxx-knl reduction-upcxx-skl \
         reduction-upcxx-openmp-knl reduction-upcxx-openmp-skl

# SKL, UPCXX (4 processes)
((run_upcxx_skl)) && { 
    printf 'Size,Time[s],Throughput[GB/s]\n'
    for i in {15..30}; do
        srun -w mp-media1 upcxx-run -n 4 -shared-heap 80% \
            reduction/reduction-upcxx-skl --size "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../reduction-shared-skl-upcxx.csv

# KNL, UPCXX (64x4 processes)
((run_upcxx_knl)) && {
    printf 'Size,Time[s],Throughput[GB/s]\n'
    for i in {15..30}; do
        srun -w mp-knl1 upcxx-run -n 64 -shared-heap 80% \
            reduction/reduction-upcxx-knl --size "$((1<<i))" --iterations "$iterations" --bench
    done 
} > ../reduction-shared-knl-upcxx.csv

# SKL, UPCXX + OpenMP (1 process, 4 threads)
((run_openmp_skl)) && {
    printf 'Size,Time[s],Throughput[GB/s]\n'
    for i in {15..30}; do
        srun -w mp-media1 upcxx-run -n 1 -shared-heap 80% \
            env OMP_NUM_THREADS=4 reduction/reduction-upcxx-openmp-skl --size "$((1<<i))" --iterations "$iterations" --bench
    done 
} > ../reduction-shared-skl-upcxx-openmp.csv

# KNL, UPCXX + OpenMP (1 process, 64 threads)
((run_openmp_knl)) && {
    printf 'Size,Time[s],Throughput[GB/s]\n'
    for i in {15..30}; do
        srun -w mp-knl1 upcxx-run -n 1 -shared-heap 80% \
            env OMP_NUM_THREADS=64 reduction/reduction-upcxx-openmp-knl --size "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../reduction-shared-knl-upcxx-openmp.csv


# ---------------------------------------
# DISTRIBUTED
# ---------------------------------------
cd -
cd build-dist
UPCXX_NETWORK=udp cmake -DCMAKE_BUILD_TYPE=Release ../..
ninja -v reduction-upcxx-knl reduction-upcxx-skl \
         reduction-upcxx-openmp-knl reduction-upcxx-openmp-skl

# SKL, UPCXX (16 processes)
((run_upcxx_skl_dist)) && {
    printf 'Size,Time[s],Throughput[GB/s]\n'
    for i in {15..30}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-media[1-4] -n %N %C" upcxx-run -N 4 -n 16 -shared-heap 80% \
            reduction/reduction-upcxx-skl --size "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../reduction-dist-skl-upcxx.csv

# KNL, UPCXX (256 processes)
((run_upcxx_knl_dist)) && {
    printf 'Size,Time[s],Throughput[GB/s]\n'
    for i in {15..30}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-knl[1-4] -n %N %C" upcxx-run -N 4 -n 256 \
            reduction/reduction-upcxx-knl --size "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../reduction-dist-knl-upcxx.csv

# SKL, UPCXX + OpenMP (4 processes, 4x4 threads)
((run_openmp_skl_dist)) && {
    printf 'Size,Time[s],Throughput[GB/s]\n'
    for i in {15..30}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-media[1-4] -n %N %C" upcxx-run -N 4 -n 4 -shared-heap 80% \
            env OMP_NUM_THREADS=4 reduction/reduction-upcxx-openmp-skl --size "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../reduction-dist-skl-upcxx-openmp.csv

# KNL, UPCXX + OpenMP (4 processes, 64x4 threads)
((run_openmp_knl_dist)) && {
    printf 'Size,Time[s],Throughput[GB/s]\n'
    for i in {15..30}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-knl[1-4] -n %N %C" upcxx-run -N 4 -n 4 -shared-heap 80% \
            env OMP_NUM_THREADS=64 reduction/reduction-upcxx-openmp-knl --size "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../reduction-dist-knl-upcxx-openmp.csv
