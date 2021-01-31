#!/bin/bash
set -e

type g++
type upcxx-run
type upcxx
gpp_flags=(-Wall -Wextra -Wpedantic -O3 -std=c++17)

sizes=()
for k in {15..30}; do
    i=$((1 << k))
    sizes+=("$i")
done

export TIMEFORMAT=$'real %3lR\tuser %3lU\tsys %3lS'

# Throughput: 4 * N * 1e-9 (GB) / Time (s)
bc_throughput() {
    local size=$1 time=$2
    bc -l <<< "4 * $size / (10^9) / $time"
}

# Enabled benchmarks
run_serial_media=1
run_numa_media=1
run_upcxx_media=1
run_serial_knl=1
run_numa_knl=1
run_upcxx_knl=1
run_upcxx_media_cluster=1
run_upcxx_knl_cluster=1

# ---------------------------------------
# SHARED MEMORY, MEDIA
# ---------------------------------------

# serial, mp-media1 (processes: 1)
if (( run_serial_media )); then
    srv='mp-media1'
    progn=reduction-serial-$srv
    g++ "${gpp_flags[@]}" -march=skylake serial.cpp -o "$progn"

    printf 'Size,Time[s],Throughput[GB/s]' > "$progn"-1.csv
    for n in "${sizes[@]}"; do
        printf >&2 'Benchmarking %s (%s rank, %s data size)\n' "$progn" 1 "$n"

        seconds=$(time srun -w "$srv" ./"$progn" --size "$n" --bench) # `time` writes to stderr
        throughput=$(bc_throughput "$n" "$seconds")
        printf '%s,%s,%s\n' "$n" "$seconds" "$throughput"
        printf >&2 '\n'
    done >> "$progn"-1.csv
fi


# NUMA, mp-media1 (OMP_NUM_THREADS: 2, 4, 8)
if (( run_numa_media )); then
    srv='mp-media1'
    progn=reduction-numa-$srv
    g++ "${gpp_flags[@]}" -march=skylake -fopenmp numa.cpp -o "$progn"

    for nproc in 2 4 8; do
        printf 'Size,Time[s],Throughput[GB/s]' > "$progn-$nproc".csv
        for n in "${sizes[@]}"; do
            printf >&2 'Benchmarking %s (%s rank, %s data size)\n' "$progn" "$nproc" "$n"

            seconds=$(time srun -w "$srv" env OMP_NUM_THREADS=$nproc ./"$progn" --size "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")
            printf '%s,%s,%s' "$n" "$seconds" "$throughput"
            printf >&2 '\n'
        done >> "$progn-$nproc".csv
    done
fi

# UPCXX, mp-media1 (UPCXX_NETWORK=smp, processes: 2, 4)
if (( run_upcxx_media )); then
    srv='mp-media1'
    progn=reduction-upcxx-$srv
    UPCXX_NETWORK=smp upcxx-run "${gpp_flags[@]}" -march=skylake upcxx.cpp -o "$progn"

    for nproc in 2 4; do
        printf 'Size,Time[s],Throughput[GB/s]' > "$progn-$nproc".csv
        for n in "${sizes[@]}"; do
            printf >&2 'Benchmarking %s (%s rank, %s data size)\n' "$progn" "$nproc" "$n"
            
            seconds=$(time srun -w "$srv" upcxx-run -n "$nproc" -shared-heap 50% ./"$progn" --size "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")
            printf '%s,%s,%s' "$n" "$seconds" "$throughput"
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
    progn=reduction-serial-$srv
    g++ "${gpp_flags[@]}" -march=knl serial.cpp -o "$progn"

    printf 'Size,Time[s],Throughput[GB/s]' > "$progn"-1.csv
    for n in "${sizes[@]}"; do
        printf >&2 'Benchmarking %s (%s rank, %s data size)\n' "$progn" 1 "$n"
        
        seconds=$(time srun -w "$srv" ./"$progn" --size "$n" --bench)
        throughput=$(bc_throughput "$n" "$seconds")
        printf '%s,%s,%s' "$n" "$seconds" "$throughput"
        printf >&2 '\n'
    done >> "$progn"-1.csv
fi

# NUMA, mp-knl1 (OMP_NUM_THREADS: 2, 4, 8, 16, 32, 64)
if (( run_numa_knl )); then
    srv='mp-knl1'
    progn=reduction-numa-$srv
    g++ "${gpp_flags[@]}" -march=knl -fopenmp numa.cpp -o "$progn"

    for nproc in 2 4 8 16 32 64; do
        printf 'Size,Time[s],Throughput[GB/s]' > "$progn-$nproc".csv
        for n in "${sizes[@]}"; do
            printf >&2 'Benchmarking %s (%s rank, %s data size)\n' "$progn" "$nproc" "$n"
            
            seconds=$(time srun -w "$srv" env OMP_NUM_THREADS=$nproc ./"$progn" --size "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")
            printf '%s,%s,%s' "$n" "$seconds" "$throughput"
            printf >&2 '\n'
        done >> "$progn-$nproc".csv
    done
fi

# UPCXX, mp-knl1 (UPCXX_NETWORK=smp, processes: 2, 4, 8, 16, 32, 64)
if (( run_upcxx_knl )); then
    srv='mp-knl1'
    progn=reduction-upcxx-$srv
    UPCXX_NETWORK=smp upcxx-run "${gpp_flags[@]}" -march=knl upcxx.cpp -o "$progn"

    for nproc in 2 4 8 16 32 64; do
        printf 'Size,Time[s],Throughput[GB/s]' > "$progn-$nproc".csv
        for n in "${sizes[@]}"; do
            printf >&2 'Benchmarking %s (%s rank, %s data size)\n' "$progn" "$nproc" "$n"
            
            seconds=$(time upcxx-run -n "$nproc" -shared-heap 50% ./"$progn" --size "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")
            printf '%s,%s,%s' "$n" "$seconds" "$throughput"
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
    progn=reduction-upcxx-$srv
    UPCXX_NETWORK=udp upcxx-run "${gpp_flags[@]}" -march=skylake upcxx.cpp -o "$progn"

    for nproc in 4 8 16; do
        printf 'Size,Time[s],Throughput[GB/s]' > "$progn-$nproc".csv
        for n in "${sizes[@]}"; do
            printf >&2 'Benchmarking %s (%s rank, %s data size)\n' "$progn" "$nproc" "$n"
            
            seconds=$(time GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w $srv -n %N %C" \
                upcxx-run -N 4 -n "$nproc" -shared-heap 50% ./"$progn" --size "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")
            printf '%s,%s,%s' "$n" "$seconds" "$throughput"
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
    progn=reduction-upcxx-$srv
    UPCXX_NETWORK=udp upcxx-run "${gpp_flags[@]}" -march=knl upcxx.cpp -o "$progn"

    for nproc in 4 8 16 32 64; do
        printf 'Size,Time[s],Throughput[GB/s]' > "$progn-$nproc".csv
        for n in "${sizes[@]}"; do
            printf >&2 'Benchmarking %s (%s rank, %s data size)\n' "$progn" "$nproc" "$n"
            
            seconds=$(time GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w $srv -n %N %C" \
                upcxx-run -N 4 -n "$nproc" -shared-heap 50% ./"$progn" --size "$n" --bench)
            throughput=$(bc_throughput "$n" "$seconds")
            printf '%s,%s,%s' "$n" "$seconds" "$throughput"
            printf >&2 '\n'
        done >> "$progn-$nproc".csv
    done
fi