#!/bin/bash
set -e

type g++
type upcxx-run
type upcxx
gpp_flags=(-Wall -Wextra -Wpedantic -O3 -std=c++17 -march=native)

dims=()
for k in {5..14}; do
    i=$((1 << k))
    dims+=("$i")
done

g++ "${gpp_flags[@]}" serial.cpp -o test-serial
g++ "${gpp_flags[@]}" -fopenmp openmp.cpp -o test-openmp
upcxx "${gpp_flags[@]}" upcxx.cpp -o test-upcxx

seed=42
num_repeats=10 # multiple checks (race conditions)
for n in "${dims[@]}"; do
    ./test-serial --dim "$n" --seed="$seed" --write

    for i in $(seq 1 "$num_repeats"); do
        printf >&2 'Testing parallel vs. serial symmetrization, dimension %s, iteration %s\n' "${n}x${n}" "$i"
        OMP_NUM_THREADS=8 ./test-openmp --dim "$n" --seed="$seed" --write
        diff -q 'serial_matrix.txt' 'openmp_matrix.txt' # assumes implicit rounding by output stream
        diff -q 'serial_matrix_symmetrized.txt' 'openmp_matrix_symmetrized.txt'

        upcxx-run -n 8 -shared-heap 50% ./test-upcxx --dim "$n" --seed="$seed" --write
        diff -q 'serial_matrix.txt' 'upcxx_matrix.txt'
        diff -q 'serial_matrix_symmetrized.txt' 'upcxx_matrix_symmetrized.txt'
    done
done
