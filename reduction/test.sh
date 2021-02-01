#!/bin/bash
set -e

type g++
type upcxx-run
type upcxx
gpp_flags=(-Wall -Wextra -Wpedantic -g -std=c++17 -march=native)

sizes=()
for k in {15..30}; do
    i=$((1 << k))
    sizes+=("$i")
done

(set -x; g++ "${gpp_flags[@]}" serial.cpp -o test-serial)
(set -x; g++ "${gpp_flags[@]}" -fopenmp openmp.cpp -o test-openmp)
(set -x; upcxx "${gpp_flags[@]}" upcxx.cpp -o test-upcxx)

seed=42
num_repeats=10 # multiple checks (race conditions)
for n in "${sizes[@]}"; do
    sum_1=$(./test-serial --size "$n" --seed="$seed" --write)

    for i in $(seq 1 "$num_repeats"); do
        printf >&2 'Testing parallel vs. serial reduction, size %s, iteration %s\n' "$n" "$i"
        sum_2=$(OMP_NUM_THREADS=8 ./test-openmp --size "$n" --seed="$seed" --write)
        sum_3=$(upcxx-run -n 8 -shared-heap 50% ./test-upcxx --size "$n" --seed="$seed" --write)

        [[ $sum_2 == $sum_1 ]] || { # assumes implicit rounding by output stream
            printf >&2 'inegality for size %s, seed %s, iteration %s\n' "$n" "$seed" "$i"
            printf >&2 'result of serial reduction: %s' "$sum_1"
            printf >&2 'result of openmp reduction: %s' "$sum_2"
            exit 1
        }
        [[ $sum_3 == $sum_1 ]] || {
            printf >&2 'inegality for size %s, seed %s, iteration %s\n' "$n" "$seed" "$i"
            printf >&2 'result of serial reduction: %s' "$sum_1"
            printf >&2 'result of upcxx reduction: %s' "$sum_3"
            exit 1
        }
    done
done
