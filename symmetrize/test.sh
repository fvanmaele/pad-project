#!/bin/bash
set -e
type upcxx-run
type cmake
type ninja

num_repeats=10

cmake() {
    command cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE="$HOME/source/vcpkg/scripts/buildsystems/vcpkg.cmake" "$@"
}

[[ -d build-test ]] && rm -rf build-test
mkdir build-test
cd build-test

UPCXX_NETWORK=smp cmake -DCMAKE_BUILD_TYPE=Release ../..
ninja -v symmetrize symmetrize-upcxx symmetrize-upcxx-openmp

for exp in {5..14}; do
    dim=$((1<<exp))
    symmetrize/symmetrize --dim "$dim" --write

    for i in $(seq 1 "$num_repeats"); do
        printf >&2 'symmetrize-upcxx, dimension %d, iteration %d\n' "$dim" "$i"
        upcxx-run -n 4 -shared-heap 50% \
            symmetrize/symmetrize-upcxx --dim "$dim" --write

        diff -q 'serial_matrix.txt' 'upcxx_matrix.txt'
        diff -q 'serial_matrix_symmetrized.txt' 'upcxx_matrix_symmetrized.txt'

        printf >&2 'symmetrize-upcxx-openmp, dimension %d, iteration %d\n' "$dim" "$i"
        upcxx-run -n 4 -shared-heap 50% \
            env OMP_NUM_THREADS=4 symmetrize/symmetrize-upcxx-openmp --dim "$dim" --write

        diff -q 'serial_matrix.txt' 'openmp_matrix.txt' # assumes implicit rounding by output stream
        diff -q 'serial_matrix_symmetrized.txt' 'openmp_matrix_symmetrized.txt'
    done
done
