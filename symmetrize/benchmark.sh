#!/bin/bash
set -eu
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
iterations=10

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
ninja -v symmetrize-upcxx-knl symmetrize-upcxx-skl \
         symmetrize-upcxx-openmp-knl symmetrize-upcxx-openmp-skl


# SKL, UPCXX (4 processes)
((run_upcxx_skl)) && { 
    printf 'X,Time[s],Throughput[GB/s]\n'
    
    for i in {5..14}; do
        srun -w mp-media1 upcxx-run -n 4 -shared-heap 80% \
            symmetrize/symmetrize-upcxx-skl --dim "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../symmetrize-shared-skl-upcxx.csv


# KNL, UPCXX (max. 64 processes)
((run_upcxx_knl)) && {
    nproc_min=8
    printf 'X,Time[s],Throughput[GB/s]\n'

    for i in {5..7}; do    
        srun -w mp-knl1 upcxx-run -n "$nproc_min" -shared-heap 80% \
            symmetrize/symmetrize-upcxx-knl --dim "$((1<<i))" --iterations "$iterations" --bench
        nproc_min=$((nproc_min * 2))
    done

    for i in {8..14}; do
        srun -w mp-knl1 upcxx-run -n 64 -shared-heap 80% \
            symmetrize/symmetrize-upcxx-knl --dim "$((1<<i))" --iterations "$iterations" --bench
    done 
} > ../symmetrize-shared-knl-upcxx.csv


# SKL, UPCXX + OpenMP (1 process, 4 threads)
((run_openmp_skl)) && {
    printf 'X,Time[s],Throughput[GB/s]\n'
    
    for i in {5..14}; do
        srun -w mp-media1 upcxx-run -n 1 -shared-heap 80% env OMP_PLACES=cores OMP_PROC_BIND=true OMP_NUM_THREADS=4 \                
            symmetrize/symmetrize-upcxx-openmp-skl --dim "$((1<<i))" --iterations "$iterations" --bench
    done 
} > ../symmetrize-shared-skl-upcxx-openmp.csv


# KNL, UPCXX + OpenMP (1 process, max. 64 threads)
((run_openmp_knl)) && {
    nproc_min=8
    printf 'X,Time[s],Throughput[GB/s]\n'

    for i in {5..7}; do
        srun -w mp-knl1 upcxx-run -n 1 -shared-heap 80% env OMP_PLACES=cores OMP_PROC_BIND=true OMP_NUM_THREADS="$nproc_min" \
            symmetrize/symmetrize-upcxx-openmp-knl --dim "$((1<<i))" --iterations "$iterations" --bench
        nproc_min=$((nproc_min * 2))
    done

    for i in {8..14}; do
        srun -w mp-knl1 upcxx-run -n 1 -shared-heap 80% env OMP_PLACES=cores OMP_PROC_BIND=true OMP_NUM_THREADS=64 \
            symmetrize/symmetrize-upcxx-openmp-knl --dim "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../symmetrize-shared-knl-upcxx-openmp.csv


# ---------------------------------------
# DISTRIBUTED
# ---------------------------------------
cd -
cd build-dist
UPCXX_NETWORK=udp cmake -DCMAKE_BUILD_TYPE=Release ../..
ninja -v symmetrize-upcxx-knl symmetrize-upcxx-skl \
         symmetrize-upcxx-openmp-knl symmetrize-upcxx-openmp-skl


# SKL, UPCXX (max. 16 processes)
((run_upcxx_skl_dist)) && {
    printf 'X,Time[s],Throughput[GB/s]\n'
    
    GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-media[1-4] -n %N %C" upcxx-run -N 4 -n 8 -shared-heap 80% \
        symmetrize/symmetrize-upcxx-skl --dim "$((1<<5))" --iterations "$iterations" --bench

    for i in {6..14}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-media[1-4] -n %N %C" upcxx-run -N 4 -n 16 -shared-heap 80% \
            symmetrize/symmetrize-upcxx-skl --dim "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../symmetrize-dist-skl-upcxx.csv


# KNL, UPCXX (max. 256 processes)
((run_upcxx_knl_dist)) && {
    nproc_min=8
    printf 'X,Time[s],Throughput[GB/s]\n'

    for i in {5..9}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-knl[1-4] -n %N %C" upcxx-run -N 4 -n "$nproc_min" \
            symmetrize/symmetrize-upcxx-knl --dim "$((1<<i))" --iterations "$iterations" --bench
        nproc_min=$((nproc_min * 2))
    done
        
    for i in {10..14}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-knl[1-4] -n %N %C" upcxx-run -N 4 -n 256 \
            symmetrize/symmetrize-upcxx-knl --dim "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../symmetrize-dist-knl-upcxx.csv


# SKL, UPCXX + OpenMP (4 processes, max. 16 threads)
((run_openmp_skl_dist)) && {
    printf 'X,Time[s],Throughput[GB/s]\n'

    GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-media[1-4] -n %N %C" upcxx-run -N 4 -n 4 -shared-heap 80% \
        env OMP_PLACES=cores OMP_PROC_BIND=true OMP_NUM_THREADS=2 symmetrize/symmetrize-upcxx-openmp-skl --dim "$((1<<5))" --iterations "$iterations" --bench
    
    for i in {6..14}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-media[1-4] -n %N %C" upcxx-run -N 4 -n 4 -shared-heap 80% \
            env OMP_PLACES=cores OMP_PROC_BIND=true OMP_NUM_THREADS=4 symmetrize/symmetrize-upcxx-openmp-skl --dim "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../symmetrize-dist-skl-upcxx-openmp.csv


# KNL, UPCXX + OpenMP (4 processes, max. 256 threads)
((run_openmp_knl_dist)) && {
    nproc_min=2
    printf 'X,Time[s],Throughput[GB/s]\n'

    for i in {5..9}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-knl[1-4] -n %N %C" upcxx-run -N 4 -n 4 -shared-heap 80% \
            env OMP_PLACES=cores OMP_PROC_BIND=true OMP_NUM_THREADS="$nproc_min" symmetrize/symmetrize-upcxx-openmp-knl --dim "$((1<<i))" --iterations "$iterations" --bench
        nproc_min=$((nproc_min * 2))
    done

    for i in {10..14}; do
        GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-knl[1-4] -n %N %C" upcxx-run -N 4 -n 4 -shared-heap 80% \
            env OMP_PLACES=cores OMP_PROC_BIND=true OMP_NUM_THREADS=64 symmetrize/symmetrize-upcxx-openmp-knl --dim "$((1<<i))" --iterations "$iterations" --bench
    done
} > ../symmetrize-dist-knl-upcxx-openmp.csv
