#!/bin/bash
set -eu
min=32
max=512
radius=4 # default values from sample benchmark script
steps=5

cmake() {
    command cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE='/home/user/source/vcpkg/scripts/buildsystems/vcpkg.cmake'
}

bench() {
    local progn=$1 min=$2 max=$3 radius=$4 steps=$5
    local x=$min 
    local y=$min 
    local z=$min
    printf 'X,Y,Z,Timesteps,Radius,Time[s],Throughput[GB/s]\n'
    
    # Alternate the doubling of the x-, y- and z-dimension.
    while (( x < max )); do
        printf >&2 'Benchmarking x=%d, y=%d, z=%d, radius=%d, steps=%d\n' "$x" "$y" "$z" "$radius" "$steps"
        "$progn" -x "$x" -y "$y" -z "$z" --radius "$radius" --steps "$steps" --bench
        x=$((x * 2))
        
        printf >&2 'Benchmarking x=%d, y=%d, z=%d, radius=%d, steps=%d\n' "$x" "$y" "$z" "$radius" "$steps"
        "$progn" -x "$x" -y "$y" -z "$z" --radius "$radius" --steps "$steps" --bench
        y=$((y * 2))
        
        printf >&2 'Benchmarking x=%d, y=%d, z=%d, radius=%d, steps=%d\n' "$x" "$y" "$z" "$radius" "$steps"
        "$progn" -x "$x" -y "$y" -z "$z" --radius "$radius" --steps "$steps" --bench
        z=$((z * 2))
    done

    printf >&2 'Benchmarking x=%d, y=%d, z=%d, radius=%d, steps=%d\n' "$x" "$y" "$z" "$radius" "$steps"
    "$progn" -x "$x" -y "$y" -z "$z" --radius "$radius" --steps "$steps" --bench
}

rm -r build-shared
mkdir build-shared
rm -r build-dist
mkdir build-dist

cd build-shared
# XXX: this uses the top-level CMakeLists; use standalone CMakeLists for reduction/symmetrize/stencil
UPCXX_NETWORK=smp cmake -DCMAKE_BUILD_TYPE=Release ../.. # -march=knl, -march=skylake, smp conduit
while (( x < max )); do
    bench stencil/upcxx-stencil-skl "$min" "$max" "$radius" "$steps" > stencil-shared-skl.csv
    bench stencil/upcxx-stencil-knl "$min" "$max" "$radius" "$steps" > stencil-shared-knl.csv
done

cd -
cd build-dist
# XXX: this uses the top-level CMakeLists; use standalone CMakeLists for reduction/symmetrize/stencil
UPCXX_NETWORK=udp cmake -DCMAKE_BUILD_TYPE=Release ../.. # -march=knl, -march=skylake, udp conduit
while (( x < max )); do
    bench stencil/upcxx-stencil-skl "$min" "$max" "$radius" "$steps" > stencil-dist-skl.csv
    bench stencil/upcxx-stencil-knl "$min" "$max" "$radius" "$steps" > stencil-dist-knl.csv
done