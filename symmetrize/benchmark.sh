#!/bin/bash
set -e

type g++
type upcxx-run
type upcxx
gpp_flags=(-Wall -Wextra -Wpedantic -O3 -std=c++17)

dims=({5..14})
export TIMEFORMAT=$'real %3lR\tuser %3lU\tsys %3lS'

# Throughput: 2 * dim * (dim-1) * sizeof(float) * 1e-9 / time
bc_throughput() {
    local dim=$1 time=$2
    bc -l <<< "2 * $dim * ($dim - 1) * 4 * (10^-9) / $time"
}

# Enabled benchmarks
run_serial_media=1
run_openmp_media=1
run_upcxx_media=1
run_serial_knl=1
run_openmp_knl=1
run_upcxx_knl=1
run_upcxx_media_cluster=1
run_upcxx_knl_cluster=1

# ---------------------------------------
# SHARED MEMORY, MEDIA
# ---------------------------------------

# serial, mp-media1 (processes: 1)
if (( run_serial_media )); then
    srv='mp-media1'
    progn=symmetrize-skl-serial
    (set -x; g++ "${gpp_flags[@]}" -march=skylake serial.cpp -o "$progn")

    printf 'X,Time[s],Throughput[GB/s]\n' > "$progn".csv
    for exp in "${dims[@]}"; do
        n=$((1 << exp))
        printf >&2 'Benchmarking %s (rank %s, dimension %s)\n' "$progn" 1 "${n}x${n}"
        seconds=$(time srun -w "$srv" ./"$progn" --dim "$n" --bench) # `time` writes to stderr
        throughput=$(bc_throughput "$n" "$seconds")

        printf '%s,%s,%s\n' "$exp" "$seconds" "$throughput"
        printf >&2 '\n'
    done >> "$progn".csv
fi


# OpenMP, mp-media1 (OMP_NUM_THREADS: 2, 4, 8)
if (( run_openmp_media )); then
    srv='mp-media1'
    progn=symmetrize-skl-shared-openmp
    (set -x; g++ "${gpp_flags[@]}" -march=skylake -fopenmp openmp.cpp -o "$progn")

    for nproc in 2 4 8; do
        printf 'X,Time[s],Throughput[GB/s]\n' > "$progn-$nproc".csv
        for exp in "${dims[@]}"; do
            n=$((1 << exp))
            if ! (( n * (n-1) / 2 / nproc % 2 == 0 )); then
                continue
            fi
            printf >&2 'Benchmarking %s (rank %s, dimension %s)\n' "$progn" "$nproc" "${n}x${n}"
            seconds=$(time srun -w "$srv" env OMP_NUM_THREADS=$nproc ./"$progn" --dim "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")

            printf '%s,%s,%s\n' "$exp" "$seconds" "$throughput"
            printf >&2 '\n'
        done >> "$progn-$nproc".csv
    done
fi

# UPCXX, mp-media1 (UPCXX_NETWORK=smp, processes: 2, 4, 8)
if (( run_upcxx_media )); then
    srv='mp-media1'
    progn=symmetrize-skl-shared-upcxx
    (set -x; UPCXX_NETWORK=smp upcxx "${gpp_flags[@]}" -march=skylake upcxx.cpp -o "$progn")

    for nproc in 2 4 8; do
        printf 'X,Time[s],Throughput[GB/s]\n' > "$progn-$nproc".csv
        for exp in "${dims[@]}"; do
            n=$((1 << exp))
            if ! (( n * (n-1) / 2 / nproc % 2 == 0 )); then
                continue
            fi
            printf >&2 'Benchmarking %s (rank %s, dimension %s)\n' "$progn" "$nproc" "${n}x${n}"
            seconds=$(time srun -w "$srv" upcxx-run -n "$nproc" -shared-heap 50% ./"$progn" --dim "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")

            printf '%s,%s,%s\n' "$exp" "$seconds" "$throughput"
            printf >&2 '\n'
        done >> "$progn-$nproc".csv
    done
fi

# ---------------------------------------
# SHARED MEMORY, KNL
# ---------------------------------------

# serial, mp-knl1 (processes: 1)
if (( run_serial_knl )); then
    srv='mp-knl1'
    progn=symmetrize-knl-serial
    (set -x; g++ "${gpp_flags[@]}" -march=knl serial.cpp -o "$progn")

    printf 'X,Time[s],Throughput[GB/s]\n' > "$progn".csv
    for exp in "${dims[@]}"; do
        n=$((1 << exp))
        printf >&2 'Benchmarking %s (rank %s, dimension %s)\n' "$progn" 1 "${n}x${n}"
        seconds=$(time srun -w "$srv" ./"$progn" --dim "$n" --bench)
        throughput=$(bc_throughput "$n" "$seconds")

        printf '%s,%s,%s\n' "$exp" "$seconds" "$throughput"
        printf >&2 '\n'
    done >> "$progn".csv
fi

# OpenMP, mp-knl1 (OMP_NUM_THREADS: 2, 4, 8, 16, 32, 64)
if (( run_openmp_knl )); then
    srv='mp-knl1'
    progn=symmetrize-knl-shared-openmp
    (set -x; g++ "${gpp_flags[@]}" -march=knl -fopenmp openmp.cpp -o "$progn")

    for nproc in 2 4 8 16 32 64; do
        printf 'X,Time[s],Throughput[GB/s]\n' > "$progn-$nproc".csv
        for exp in "${dims[@]}"; do
            n=$((1 << exp))
            if ! (( n * (n-1) / 2 / nproc % 2 == 0 )); then
                continue
            fi
            printf >&2 'Benchmarking %s (rank %s, dimension %s)\n' "$progn" "$nproc" "${n}x${n}"
            seconds=$(time srun -w "$srv" env OMP_NUM_THREADS=$nproc ./"$progn" --dim "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")

            printf '%s,%s,%s\n' "$exp" "$seconds" "$throughput"
            printf >&2 '\n'
        done >> "$progn-$nproc".csv
    done
fi

# UPCXX, mp-knl1 (UPCXX_NETWORK=smp, processes: 2, 4, 8, 16, 32, 64)
if (( run_upcxx_knl )); then
    srv='mp-knl1'
    progn=symmetrize-knl-shared-upcxx
    (set -x; UPCXX_NETWORK=smp upcxx "${gpp_flags[@]}" -march=knl upcxx.cpp -o "$progn")

    for nproc in 2 4 8 16 32 64; do
        printf 'X,Time[s],Throughput[GB/s]\n' > "$progn-$nproc".csv
        for exp in "${dims[@]}"; do
            n=$((1 << exp))
            if ! (( n * (n-1) / 2 / nproc % 2 == 0 )); then
                continue
            fi
            printf >&2 'Benchmarking %s (rank %s, dimension %s)\n' "$progn" "$nproc" "${n}x${n}"
            seconds=$(time srun -w "$srv" upcxx-run -n "$nproc" -shared-heap 50% ./"$progn" --dim "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")

            printf '%s,%s,%s\n' "$exp" "$seconds" "$throughput"
            printf >&2 '\n'
        done >> "$progn-$nproc".csv
    done
fi

# ---------------------------------------
# DISTRIBUTED, MEDIA
# ---------------------------------------

# UPCXX, mp-media[1-4] (UPCXX_NETWORK=udp, processes: 4, 8, 16)
if (( run_upcxx_media_cluster )); then
    srv='mp-media[1-4]'
    progn=symmetrize-skl-dist-upcxx
    (set -x; UPCXX_NETWORK=udp upcxx "${gpp_flags[@]}" -march=skylake upcxx.cpp -o "$progn")

    for nproc in 4 8 16; do
        printf 'X,Time[s],Throughput[GB/s]\n' > "$progn-$nproc".csv
        for exp in "${dims[@]}"; do
            n=$((1 << exp))
            if ! (( n * (n-1) / 2 / nproc % 2 == 0 )); then
                continue
            fi
            printf >&2 'Benchmarking %s (rank %s, dimension %s)\n' "$progn" "$nproc" "${n}x${n}"
            seconds=$(time GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w $srv -n %N %C" \
                upcxx-run -N 4 -n "$nproc" -shared-heap 50% ./"$progn" --dim "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")

            printf '%s,%s,%s\n' "$exp" "$seconds" "$throughput"
            printf >&2 '\n'
        done >> "$progn-$nproc".csv
    done
fi


# ---------------------------------------
# DISTRIBUTED, KNL
# ---------------------------------------

# UPCXX, mp-knl[1-4] (UPCXX_NETWORK=udp, processes: 4, 8, 16, 32, 64)
if (( run_upcxx_knl_cluster )); then
    srv='mp-knl[1-4]'
    progn=symmetrize-knl-dist-upcxx
    (set -x; UPCXX_NETWORK=udp upcxx "${gpp_flags[@]}" -march=knl upcxx.cpp -o "$progn")

    for nproc in 4 8 16 32 64; do
        printf 'X,Time[s],Throughput[GB/s]\n' > "$progn-$nproc".csv
        for exp in "${dims[@]}"; do
            n=$((1 << exp))
            if ! (( n * (n-1) / 2 / nproc % 2 == 0 )); then
                continue
            fi
            printf >&2 'Benchmarking %s (rank %s, dimension %s)\n' "$progn" "$nproc" "${n}x${n}"
            seconds=$(time GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w $srv -n %N %C" \
                upcxx-run -N 4 -n "$nproc" -shared-heap 50% ./"$progn" --dim "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")

            printf '%s,%s,%s\n' "$exp" "$seconds" "$throughput"
            printf >&2 '\n'
        done >> "$progn-$nproc".csv
    done
fi
