#!/bin/bash
set -e

type g++
type upcxx-run
type upcxx
gpp_flags=(-Wall -Wextra -Wpedantic -O3 -std=c++17)

dims=({5..14})

export TIMEFORMAT=$'real %3lR\tuser %3lU\tsys %3lS'

# Throughput: dim * (dim-1) * sizeof(float) * 1e-9 / time
bc_throughput() {
     local dim=$1 time=$2
     bc -l <<< "$dim * ($dim - 1) * 4 / (10^9) / $time"
}
 
bc_add() { bc -l <<< "$1 + $2"; }
bc_div() { bc -l <<< "$1 / $2"; }

# Enabled benchmarks
run_serial_media=0
run_openmp_media=1
run_upcxx_media=1
run_serial_knl=0
run_openmp_knl=1
run_upcxx_knl=1
run_upcxx_media_cluster=1
run_upcxx_knl_cluster=1

# Number of iterations (averaged over)
iterations=5

# Arguments: $1 progn ${@:2} spawned process (takes --dim, --bench as arguments)
run_benchmark() {
    local progn=$1 nproc=$2 # iterations, size
    shift 2

    printf 'X,Time[s],Throughput[GB/s]\n'
    for exp in "${dims[@]}"; do
        local seconds=0 throughput= n=

        n=$((1 << exp))
        if ! (( n * (n-1) / 2 / nproc % 2 == 0 )); then
            continue
        fi
        
        for i in $(seq 1 "$iterations"); do
            printf >&2 'Benchmarking %s (rank %s, dimension %s, iteration %s)\n' "$progn" "$nproc" "$n" "$i"
            seconds_it=$(time "$@" --dim "$n" --bench) # `time` writes to stderr 
            seconds=$(bc_add "$seconds" "$seconds_it")
        done
        seconds=$(bc_div "$seconds" "$iterations")
        throughput=$(bc_throughput "$n" "$seconds")

        printf '%s,%s,%s\n' "$exp" "$seconds" "$throughput"
        printf >&2 '\n'
    done
}

# ---------------------------------------
# SHARED MEMORY, MEDIA
# ---------------------------------------

# Serial, mp-media1 (processes: 1)
if (( run_serial_media )); then
    srv=mp-media1
    progn=symmetrize-skl-serial    
    (set -x; g++ "${gpp_flags[@]}" -march=skylake serial.cpp -o "$progn")

    run_benchmark "$progn" 1 \
        srun -w "$srv" ./"$progn" > "$progn".csv
fi


# OpenMP, mp-media1 (OMP_NUM_THREADS: 2, 4, 8)
if (( run_openmp_media )); then
    srv=mp-media1
    progn=symmetrize-skl-shared-openmp
    (set -x; g++ "${gpp_flags[@]}" -march=skylake -fopenmp openmp.cpp -o "$progn")

    for nproc in 2 4 8; do
        run_benchmark "$progn" "$nproc" \
            srun -w "$srv" env OMP_NUM_THREADS=$nproc ./"$progn" > "$progn-$nproc".csv
    done
fi

# UPCXX, mp-media1 (UPCXX_NETWORK=smp, processes: 2, 4, 8)
if (( run_upcxx_media )); then
    srv=mp-media1
    progn=symmetrize-skl-shared-upcxx
    (set -x; UPCXX_NETWORK=smp upcxx "${gpp_flags[@]}" -march=skylake upcxx.cpp -o "$progn")

    for nproc in 2 4 8; do
        run_benchmark "$progn" "$nproc" \
            srun -w "$srv" upcxx-run -n "$nproc" -shared-heap 80% ./"$progn" > "$progn-$nproc".csv
    done
fi

# ---------------------------------------
# SHARED MEMORY, KNL
# ---------------------------------------

# Serial, mp-knl1 (processes: 1)
if (( run_serial_knl )); then
    srv=mp-knl1
    progn=symmetrize-knl-serial
    (set -x; g++ "${gpp_flags[@]}" -march=knl serial.cpp -o "$progn")

    run_benchmark "$progn" 1 \
        srun -w "$srv" ./"$progn" > "$progn".csv
fi

# OpenMP, mp-knl1 (OMP_NUM_THREADS: 2, 4, 8, 16, 32, 64)
if (( run_openmp_knl )); then
    srv='mp-knl1'
    progn=symmetrize-knl-shared-openmp
    (set -x; g++ "${gpp_flags[@]}" -march=knl -fopenmp openmp.cpp -o "$progn")

    for nproc in 2 4 8 16 32 64; do
        run_benchmark "$progn" "$nproc" \
            srun -w "$srv" env OMP_NUM_THREADS=$nproc ./"$progn" > "$progn-$nproc".csv
    done
fi

# UPCXX, mp-knl1 (UPCXX_NETWORK=smp, processes: 2, 4, 8, 16, 32, 64)
if (( run_upcxx_knl )); then
    srv='mp-knl1'
    progn=symmetrize-knl-shared-upcxx
    (set -x; UPCXX_NETWORK=smp upcxx "${gpp_flags[@]}" -march=knl upcxx.cpp -o "$progn")

    for nproc in 2 4 8 16 32 64; do
        run_benchmark "$progn" "$nproc" \
            srun -w "$srv" upcxx-run -n "$nproc" -shared-heap 80% ./"$progn" > "$progn-$nproc".csv
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
        run_benchmark "$progn" "$nproc" \
            env GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w $srv -n %N %C" \
                upcxx-run -N 4 -n "$nproc" -shared-heap 80% ./"$progn" > "$progn-$nproc".csv
    done
fi


# ---------------------------------------
# DISTRIBUTED, KNL
# ---------------------------------------

# UPCXX, mp-knl[1-4] (UPCXX_NETWORK=udp, processes: 4, 8, 16, 32, 64, 128, 256)
if (( run_upcxx_knl_cluster )); then
    srv='mp-knl[1-4]'
    progn=symmetrize-knl-dist-upcxx
    (set -x; UPCXX_NETWORK=udp upcxx "${gpp_flags[@]}" -march=knl upcxx.cpp -o "$progn")

    for nproc in 4 8 16 32 64 128 256; do
        run_benchmark "$progn" "$nproc" \
            env GASNET_SPAWNFN=C GASNET_CSPAWN_CMD="srun -w $srv -n %N %C" \
                upcxx-run -N 4 -n "$nproc" -shared-heap 80% ./"$progn" > "$progn-$nproc".csv
    done
fi
