- [Introduction](#introduction)
- [Comparison to serial implementation](#comparison-to-serial-implementation)
- [Parallel implementation](#parallel-implementation)
  - [Tasks](#tasks)
  - [Implementation](#implementation)
- [Benchmarks](#benchmarks)

## Introduction

In this exercise we consider an *reduction* of a `float` array, varying in size `N` from `1<<15` to `1<<30`.
As the goal is to perform the reduction on a distributed system, this is implemented in a hierarchical
fashion: partial sums are first computed on each process, then combined.

![Hierarchical reduction](https://mp-force.ziti.uni-heidelberg.de/asc/projects/lectures/parallel-algorithm-design/ws20/upcxx/-/raw/master/reduction/hierarchical_reduction.png)

## Comparison to serial implementation

In the serial implementation, the full range of values is added to a reduction value in a single iteration,
for example with `std::accumulate`. As described above, a parallel implementation adds a smaller range of values, and then combines the results. In particular, it may have better numerical stability than a serial implementation. To avoid this issue, there are several approaches:

* Use a reduction value with greater precision than the input values (`double` or `long double`);
* Use [pairwise summation](https://en.wikipedia.org/wiki/Pairwise_summation);
* Use [Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm).

A deeper discussion of these methods is out of the scope of this document; see e.g. [1][ref-1] or [2][ref-2].
For simplicity, we have opted for the first approach.
```c++
std::accumulate<std::vector<float>::iterator, double>(v.begin(), v.end(), 0.0);
```

The array is filled with pseudo-random values using `std::mt19937_64` and a fixed seed. To ensure consistency with the serial implementation, the parallel implementation uses `std::mt19937_64::discard` for each process (see [Parallel implementation](#parallel-implementation).) Reduction values are then compared by printing them to standard output (implicity using rounding from `std::cout`).

## Parallel implementation

### Tasks 

For the parallel implementation, we have two main concepts of "tasks":

* UPCXX *process*, which can locally on a shared memory system or distributed on a cluster;
* OpenMP *threads*, which are local to an UPCXX process.

The main reason for combining threads with processes is that processes require *communication*. If we wish to increase the amount of parallelism, this may thus result in additional overhead, especially if processes are located on different nodes and have to communicate over the network. 

While UPCXX provides additional mechanisms to reduce this overhead (i.e. `upcxx::broadcast()` and `upcxx::local_team()`), a simple way is to use a single UPCXX process per node, and a fixed amount of OpenMP threads per process. Refer to the [Benchmarks](#benchmarks) section for details.

**Important:** upcxx collectives (e.g. `upcxx::barrier`, `upcxx::reduce_one`) should only be called by the "master persona" thread, i.e. the (single) thread that called `upcxx::init`. This can be done with `#pragma omp master` inside an OpenMP parallel region, or by using multiple OpenMP regions. 

For simplicity, we assume thread affinity holds between regions (i.e. `OMP_PROC_BIND` is set to `TRUE`) and used the latter approach:
```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=true
```

### Implementation

The implementation first divides an array of size `N` evenly between processes and threads:
```c++
std::ptrdiff_t block_size = N / upcxx::rank_n();
std::ptrdiff_t block_size_omp = block_size / omp_get_num_threads();
```
where we assume (and check) the division is without remainder. As in the serial implementation, these blocks are initialized with pseudo-random values, using `std::mt19937_64::discard()` to ensure consistency.

```c++
std::mt19937_64 rgen(seed);
rgen.discard((upcxx::rank_me() * omp_get_num_threads() + omp_get_thread_num()) * block_size_omp);

#pragma omp for schedule(static)
    for (index_t i = 0; i < block_size; ++i) {
        u[i] = 0.5 + rgen() % 100;
    }
```
or, when only using UPCXX processes:

```c++
std::mt19937_64 rgen(seed);
rgen.discard(upcxx::rank_me() * block_size);

for (index_t i = 0; i < block_size; ++i) {
    u[i] = 0.5 + rgen() % 100;
}
```
Partial sums are then computed in the usual fashion (see `upcxx.cpp` and `upcxx_openmp.cpp`). The simplest way to communicate these sums between processes is `upcxx::reduce_one`. 

```c++
double psum(0);
#pragma omp parallel for simd schedule(static) reduction(+:psum)
    for (std::ptrdiff_t = 0; i < block_size; ++i) {
        psum += u[i];
    }
double sum = upcxx::reduce_one(psum, upcxx::op_fast_add, 0).wait();
```
`upcxx::reduce_one` reduces values in a **non-deterministic order** (see *UPCXX specification*, 12.2.25) and stores the result on a single process (e.g. process `0`). If a deterministic order is wanted, manual communication with e.g. `upcxx::dist_object` and `upcxx::future` is required. For example:

```c++
// Assign partial sum to distributed object (universal name, local value)
upcxx::dist_object<double> psum_d(psum);

// Reduce partial sums in ascending order on first process
if (proc_id == 0) {
    double res(*psum_d);

    // Fetch values synchronously. To fetch asynchrously, .then() can be used
    // instead of .wait(), together with upcxx::when_all().
    for (int k = 1; k < upcxx::rank_n(); ++k) {
        double psum = psum_d.fetch(k).wait();
        res += psum;
    }
}
```

## Benchmarks

We use the following criteria for benchmarking:

* Throughput is computed as `N * sizeof(float) * 1e-9 / time`;
* Array intialization is not timed, only reduction;
* 100 iterations are performed per size `N`, with `time` taken as the average over these iterations.

Process and thread amount are chosen as follows:

| Benchmark        | Nodes | Processes |
| ---------------- | ----- | --------- |
| KNL, shared      | 1     | 64        |
| SKL, shared      | 1     | 4         |
| KNL, distributed | 4     | 256       |
| SKL, distributed | 4     | 16        |

for the UPCXX benchmarks, and:

| Benchmark        | Nodes | Processes | Threads |
| ---------------- | ----- | --------- | ------- |
| KNL, shared      | 1     | 1         | 64      |
| SKL, shared      | 1     | 1         | 4       |
| KNL, distributed | 4     | 4         | 256     |
| SKL, distributed | 4     | 4         | 16      |

for the UPCXX + OpenMP benchmarks.

![UPCXX](https://mp-force.ziti.uni-heidelberg.de/asc/projects/lectures/parallel-algorithm-design/ws20/upcxx/-/raw/master/reduction/reduction.png)

![OpenMP](https://mp-force.ziti.uni-heidelberg.de/asc/projects/lectures/parallel-algorithm-design/ws20/upcxx/-/raw/master/reduction/reduction_openmp.png)

As the plots indicate, the hybrid OpenMP implementation leads to a higher throughput, from a smaller number of processes that need to communicate on the network.

**Note:** Due to an unknown `slurm` error on the asc cluster, the UPCXX benchmark does not include sizes `1<<29` and `1<<30` for distributed KNL and 256 UPCXX processes.

[ref-1]: https://hal.archives-ouvertes.fr/hal-02265534v2/document
[ref-2]: https://www.iro.umontreal.ca/~mignotte/IFT2425/Documents/AccrateSummationMethods.pdf