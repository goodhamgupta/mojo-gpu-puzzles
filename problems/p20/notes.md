# Functional Patterns

GPU threads vs SIMD operations

GPU Device
├── Grid (your entire problem)
│   ├── Block 1 (group of threads, shared memory)
│   │   ├── Warp 1 (32 threads, lockstep execution) --> We'll learn in Part VI
│   │   │   ├── Thread 1 → Your Mojo function
│   │   │   ├── Thread 2 → Your Mojo function
│   │   │   └── ... (32 threads total)
│   │   └── Warp 2 (32 threads)
│   └── Block 2 (independent group)


- lockstep execution -> Run same operation at the same time on multiple threads in parallel
- A block can have multiple warps i.e collection of threads. Each warp is usually 32 threads.

## Fundamental patterns

- Elementwise -> Maximum parallelism and auto SIMD vectorisation
- Tiled -> Memory efficient processing with cache optimisation (eg: matmul, FMA?)
- Manual vectorisation -> moar control over ops
- Mojo vectorise -> Auto vectorisation within bounds( i guess this refers to things like the @parameter decorator?)


Eg Problem: Add two 1024-element vectors (SIZE=1024, SIMD_WIDTH=4)

Elementwise:     256 threads × 1 SIMD op   = High parallelism
Tiled:           32 threads  × 8 SIMD ops  = Cache optimization
Manual:          8 threads   × 32 SIMD ops = Maximum control
Mojo vectorize:  32 threads  × 8 SIMD ops  = Automatic safety

Performance benchmarks:

Benchmark Results (SIZE=1,048,576):
elementwise:        11.34ms  ← Maximum parallelism wins at scale
tiled:              12.04ms  ← Good balance of locality and parallelism
manual_vectorized:  15.75ms  ← Complex indexing hurts simple operations
vectorized:         13.38ms  ← Automatic optimization overhead

