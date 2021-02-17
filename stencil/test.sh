#!/bin/bash
set -e
type g++; type upcxx-run; type upcxx

min=8
max=512
seed=42
num_repeats=10 # multiple checks (race conditions)

test_stencil() {
    local stencil_args=(-x "$1" -y "$2" -z "$3" --seed="$seed" --write)
    ./stencil-serial "${stencil_args[@]}"

    for i in $(seq 1 "$num_repeats"); do
        printf >&2 'Testing dimension {%d,%d,%d}, iteration %d\n' "$1" "$2" "$3" "$i"
        upcxx-run -n 4 -shared-heap 50% ./stencil-upcxx "${stencil_args[@]}"

        diff -q 'serial_stencil.txt' 'upcxx_stencil.txt'
        diff -q 'serial_stencil_steps.txt' 'upcxx_stencil_steps.txt'
    done
}

gpp_args=(-Wall -Wextra -Wpedantic -O3 -std=c++17 -march=native)
#g++ "${gpp_args[@]}" serial.cpp -o stencil-serial
#upcxx "${gpp_args[@]}" upcxx.cpp -o stencil-upcxx

x=$min
y=$min
z=$min
while (( x < max )); do
    test_stencil "$x" "$y" "$z"
    x=$((x * 2))
    test_stencil "$x" "$y" "$z"
    y=$((y * 2))
    test_stencil "$x" "$y" "$z"
    z=$((z * 2))
done
test_stencil "$x" "$y" "$z"
