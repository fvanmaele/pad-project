#!/bin/bash
set -eu
type upcxx-run
type cmake
type ninja
[[ -r $HOME/source/vcpkg/scripts/buildsystems/vcpkg.cmake ]]

# Benchmark parameters
min=32
max=512
radius=4 # default values from sample benchmark script
steps=5
iterations=4 # TODO

# Enabled benchmarks
run_upcxx_media=1
run_upcxx_knl=1
run_upcxx_media_cluster=1
run_upcxx_knl_cluster=1

cmake() {
    command cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE="$HOME/source/vcpkg/scripts/buildsystems/vcpkg.cmake" "$@"
}

bench() {
    local x=$min 
    local y=$min 
    local z=$min
    printf 'X,Y,Z,Timesteps,Radius,Time[s],Throughput[GB/s]\n'
    
    # Alternate the doubling of the x-, y- and z-dimension.
    while (( x < max )); do
        printf >&2 'Benchmarking x=%d, y=%d, z=%d, radius=%d, steps=%d\n' "$x" "$y" "$z" "$radius" "$steps"
        time "$@" -x "$x" -y "$y" -z "$z" --radius "$radius" --steps "$steps" --bench
        x=$((x * 2))
        
        printf >&2 '\nBenchmarking x=%d, y=%d, z=%d, radius=%d, steps=%d\n' "$x" "$y" "$z" "$radius" "$steps"
        time "$@" -x "$x" -y "$y" -z "$z" --radius "$radius" --steps "$steps" --bench
        y=$((y * 2))
        
        printf >&2 '\nBenchmarking x=%d, y=%d, z=%d, radius=%d, steps=%d\n' "$x" "$y" "$z" "$radius" "$steps"
        time "$@" -x "$x" -y "$y" -z "$z" --radius "$radius" --steps "$steps" --bench
        z=$((z * 2))
    done

    printf >&2 '\nBenchmarking x=%d, y=%d, z=%d, radius=%d, steps=%d\n' "$x" "$y" "$z" "$radius" "$steps"
    "$@" -x "$x" -y "$y" -z "$z" --radius "$radius" --steps "$steps" --bench
}

rm -rf build-shared
mkdir build-shared
rm -rf build-dist
mkdir build-dist

# ---------------------------------------
# SHARED MEMORY, SKL
# ---------------------------------------
cd build-shared
# XXX: this uses the top-level CMakeLists; use standalone CMakeLists for reduction/symmetrize/stencil
UPCXX_NETWORK=smp cmake -DCMAKE_BUILD_TYPE=Release ../.. # -march=knl, -march=skylake, smp conduit
ninja -v

if (( run_upcxx_media )); then
    bench srun -w 'mp-media1' \
        upcxx-run -n 4 -shared-heap 80% stencil/stencil-upcxx-skl > stencil-shared-skl-upcxx.csv
fi

# ---------------------------------------
# SHARED MEMORY, KNL
# ---------------------------------------
if (( run_upcxx_knl )); then
    bench srun -w 'mp-knl1' \
        upcxx-run -n 4 -shared-heap 80% stencil/stencil-upcxx-knl > stencil-shared-knl-upcxx.csv
fi

# ---------------------------------------
# DISTRIBUTED, SKL
# ---------------------------------------
cd -
cd build-dist
# XXX: this uses the top-level CMakeLists; use standalone CMakeLists for reduction/symmetrize/stencil
UPCXX_NETWORK=udp cmake -DCMAKE_BUILD_TYPE=Release ../.. # -march=knl, -march=skylake, udp conduit
ninja -v

if (( run_upcxx_media_cluster )); then
    bench env GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-media[1-4] -n %N %C" \
        upcxx-run -N 4 -n 4 -shared-heap 80% stencil/stencil-upcxx-skl > stencil-dist-skl-upcxx.csv
fi

# ---------------------------------------
# DISTRIBUTED, KNL
# ---------------------------------------
if (( run_upcxx_knl_cluster )); then
    bench env GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w mp-knl[1-4] -n %N %C" \
        upcxx-run -N 4 -n 4 -shared-heap 80% stencil/stencil-upcxx-knl
fi