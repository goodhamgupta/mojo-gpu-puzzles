# 📊 Benchmarking - Performance Analysis and Optimization

## Overview

After mastering **elementwise**, **tiled**, **manual vectorization**, and **Mojo vectorize** patterns, it's time to measure their actual performance. This guide explains how to use the built-in benchmarking system in `p20.mojo` to scientifically compare these approaches and understand their performance characteristics.

**Key insight:** _Theoretical analysis is valuable, but empirical benchmarking reveals the true performance story on your specific hardware._

## Running benchmarks

To execute the comprehensive benchmark suite:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p20 --benchmark
```

  </div>
  <div class="tab-content">

```bash
pixi run p20 --benchmark
```

  </div>
</div>

Your output will show performance measurements for each pattern:

```txt
SIZE: 1024
simd_width: 4
Running P20 GPU Benchmarks...
SIMD width: 4
--------------------------------------------------------------------------------
Testing SIZE=16, TILE=4
Running elementwise_16_4
Running tiled_16_4
Running manual_vectorized_16_4
Running vectorized_16_4
--------------------------------------------------------------------------------
Testing SIZE=128, TILE=16
Running elementwise_128_16
Running tiled_128_16
Running manual_vectorized_128_16
Testing SIZE=128, TILE=16, Vectorize within tiles
Running vectorized_128_16
--------------------------------------------------------------------------------
Testing SIZE=1048576 (1M), TILE=1024
Running elementwise_1M_1024
Running tiled_1M_1024
Running manual_vectorized_1M_1024
Running vectorized_1M_1024
----------------------------------------------------------
| name                      | met (ms)           | iters |
----------------------------------------------------------
| elementwise_16_4          | 4.59953155         | 100   |
| tiled_16_4                | 3.16459014         | 100   |
| manual_vectorized_16_4    | 4.60563415         | 100   |
| vectorized_16_4           | 3.15671539         | 100   |
| elementwise_128_16        | 3.1611135375       | 80    |
| tiled_128_16              | 3.1669656300000004 | 100   |
| manual_vectorized_128_16  | 3.1609855625       | 80    |
| vectorized_128_16         | 3.16142578         | 100   |
| elementwise_1M_1024       | 11.338706742857143 | 70    |
| tiled_1M_1024             | 12.044989871428571 | 70    |
| manual_vectorized_1M_1024 | 15.749412314285713 | 70    |
| vectorized_1M_1024        | 13.377229          | 100   |
----------------------------------------------------------

Benchmarks completed!
```

## Benchmark configuration

The benchmarking system uses Mojo's built-in `benchmark` module with carefully chosen parameters:

```mojo
bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
```

**Configuration explanation:**
- **`max_iters=10`**: Runs each benchmark up to 10 times for statistical reliability
- **`min_warmuptime_secs=0.2`**: Ensures GPU is warmed up before measurement begins
- **Automatic calibration**: Framework automatically determines optimal iteration count

## Benchmarking implementation deep dive

### 1. **Mojo benchmark module architecture**

The benchmarking system relies on Mojo's built-in `benchmark` module:

```mojo
from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    keep
)
```

**Key components:**
- **`Bench`**: The main benchmarking orchestrator that manages multiple benchmark runs
- **`BenchConfig`**: Specifies parameters like iteration counts and warmup time
- **`Bencher`**: Executes individual benchmark iterations and collects timing data
- **`BenchId`**: Provides unique names for benchmark identification and reporting
- **`keep`**: Hints the compiler to not optimize a variable away. This is important for correct benchmarking.

### 2. **Parameterized benchmark functions**

Each pattern has a dedicated benchmark function using compile-time parameterization:

```mojo
@parameter
@always_inline
fn benchmark_elementwise_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    # Implementation follows...
```

**Why parameterized functions?**
- **Compile-time specialization**: Each `test_size` and `tile_size` combination generates optimized code
- **Zero abstraction cost**: No runtime parameter overhead
- **Type safety**: Ensures consistent parameter usage across benchmarks
- **Performance isolation**: Each combination is independently optimized

### 3. **Workflow pattern implementation**

Each benchmark follows a consistent nested function pattern:

```mojo
@parameter
@always_inline
fn benchmark_pattern_parameterized[test_size: Int, tile_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn pattern_workflow(ctx: DeviceContext) raises:
        # 1. Setup phase: Create buffers and tensors
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        # 2. Data initialization phase
        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = 2 * i
                b_host[i] = 2 * i + 1

        # 3. Tensor creation phase
        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())

        # 4. Actual computation being benchmarked
        pattern_function[layout, dtype, SIMD_WIDTH, rank, test_size](
            out_tensor, a_tensor, b_tensor, ctx
        )

        # 5. Prevent compiler optimization - Critical for accurate benchmarking!
        keep(out.unsafe_ptr())

        # 6. Synchronization to ensure completion
        ctx.synchronize()

    # 7. Benchmark execution
    bench_ctx = DeviceContext()
    b.iter_custom[pattern_workflow](bench_ctx)
```

**Workflow phases explained:**

#### **Setup phase (steps 1-3):**
- **Buffer allocation**: `ctx.enqueue_create_buffer[dtype](test_size)` allocates GPU memory
- **Initialization**: `enqueue_fill(0)` zeros out the memory
- **Data preparation**: Maps to host memory for data initialization
- **Tensor wrapping**: Creates `LayoutTensor` views for the algorithms

#### **Computation phase (step 4):**
- **Isolated execution**: Only the algorithm computation is timed
- **Consistent parameters**: Same data layout and size across all patterns
- **GPU targeting**: All algorithms use `target="gpu"` for fair comparison

#### **Compiler optimization prevention (step 5):**
- **`keep(out.unsafe_ptr())`**: Critical function that prevents dead code elimination
- **Why it's needed**: Without `keep`, the compiler might optimize away the entire computation since the result isn't "used"
- **GPU-specific concern**: GPU kernels are asynchronous, making optimization detection more complex
- **Placement importance**: Must be after computation but before synchronization

#### **Synchronization phase (step 6):**
- **Completion guarantee**: `ctx.synchronize()` ensures GPU work completes
- **Accurate timing**: Prevents timing of incomplete GPU operations

### 4. **Custom iteration measurement**

The key to accurate GPU benchmarking is the `iter_custom` approach:

```mojo
b.iter_custom[pattern_workflow](bench_ctx)
```

**Why `iter_custom`?**
- **GPU context management**: Handles DeviceContext lifecycle properly
- **Memory management**: Ensures proper buffer cleanup between iterations
- **Synchronization**: Handles GPU-CPU synchronization automatically
- **Overhead isolation**: Separates setup cost from computation cost

**Comparison with standard iteration:**
```mojo
// Standard (problematic for GPU):
b.iter[some_function]()  // No GPU context management

// Custom (optimal for GPU):
b.iter_custom[gpu_workflow](device_context)  // Proper GPU lifecycle
```

### 5. **Benchmark orchestration**

The main benchmarking sequence follows a structured approach:

```mojo
bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
bench = Bench(bench_config)

// Register benchmarks with unique IDs
bench.bench_function[benchmark_elementwise_parameterized[16, 4]](
    BenchId("elementwise_16_4")
)
bench.bench_function[benchmark_tiled_parameterized[16, 4]](
    BenchId("tiled_16_4")
)

// Execute and report
print(bench)  // Automatic results table generation
```

**Orchestration benefits:**
- **Centralized configuration**: Single config applies to all benchmarks
- **Automatic result collection**: Framework handles timing and statistics
- **Consistent reporting**: Uniform output format across all patterns
- **Statistical reliability**: Multiple iterations with automatic outlier handling

### 6. **Memory and synchronization considerations**

#### **Critical role of `keep()` in GPU benchmarking:**
```mojo
keep(out.unsafe_ptr())  // Prevents dead code elimination
```

**The optimization problem:**
Modern compilers are aggressive about eliminating "unused" computations. In our benchmarks:
```mojo
// Without keep(), compiler might see:
pattern_function(out_tensor, a_tensor, b_tensor, ctx)
// Result never read -> entire computation eliminated!
```

**Why `keep()` is essential:**
- **Dead code elimination**: Compiler sees output buffer is never read and might skip the computation
- **GPU complexity**: Asynchronous kernel launches make optimization detection harder
- **Benchmark accuracy**: Without `keep`, you might be measuring nothing instead of the algorithm
- **Strategic placement**: Must be after computation but before synchronization to be effective

**What `keep()` does:**
- **Hints to compiler**: "This memory location contains important data, don't optimize away"
- **Zero runtime cost**: Pure compile-time hint, no performance impact
- **Prevents false results**: Ensures actual work is measured, not optimized-away code

**Real-world impact:**
Without `keep()`, all our benchmark results might show unrealistically fast times (measuring empty kernels) or even CUDA errors from eliminated GPU operations.

#### **Buffer lifecycle management:**
```mojo
// Each iteration creates fresh buffers
out = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
// Automatic cleanup when out goes out of scope
```

**Why fresh buffers per iteration?**
- **Cache neutrality**: Prevents cache warming from affecting results
- **Memory pressure testing**: Tests allocation overhead realistically
- **Consistent starting state**: Each iteration starts with identical conditions

#### **GPU synchronization strategy:**
```mojo
ctx.synchronize()  // Critical for accurate timing
```

**GPU timing challenges:**
- **Asynchronous execution**: GPU kernels launch asynchronously
- **Pipelined operations**: Multiple kernels may overlap
- **Memory transfers**: Host-device transfers add complexity

**How `synchronize()` solves this:**
- **Completion guarantee**: Ensures all GPU work finishes before timing stops
- **Accurate measurement**: Captures actual computation time, not launch time
- **Cross-platform consistency**: Works across different GPU vendors/drivers

### 7. **Parameterization strategy**

The benchmark tests specific size combinations:

```mojo
// Small problem: Tests overhead characteristics
benchmark_pattern_parameterized[16, 4]

// Medium problem: Tests scaling and cache behavior
benchmark_pattern_parameterized[128, 16]
```

**Size selection rationale:**
- **SIZE=16, TILE=4**: Minimal GPU utilization, overhead-dominated
- **SIZE=128, TILE=16**: Better GPU utilization, cache effects visible
- **Power-of-2 sizes**: Aligned with GPU memory architecture
- **Tile size relationships**: Tests different thread-to-work ratios

### 8. **Result aggregation and reporting**

The framework automatically generates the results table:

```mojo
print(bench)  // Triggers result compilation and display
```

**What happens during result compilation:**
- **Statistical analysis**: Calculates mean, best, and deviation across iterations
- **Outlier filtering**: Removes statistical outliers for more reliable results
- **Format standardization**: Converts timing to consistent units (milliseconds)
- **Tabular output**: Generates the formatted results table

**Result interpretation framework:**
- **Total execution time**: `met (ms)` shows cumulative time for all iterations
- **Iteration count**: `iters` shows how many times each benchmark ran
- **Relative comparison**: Framework enables direct pattern-to-pattern comparison

This implementation demonstrates Mojo's sophisticated approach to performance measurement, combining compile-time optimization with runtime flexibility for accurate GPU benchmarking.

## Test scenarios

The benchmark suite tests three different scenarios to reveal performance characteristics across various problem sizes:

### 1. Small problem size (SIZE=16, TILE=4)
```mojo
benchmark_elementwise_parameterized[16, 4]
benchmark_tiled_parameterized[16, 4]
benchmark_manual_vectorized_parameterized[16, 4]
benchmark_vectorized_parameterized[16, 4]
```

**Purpose:**
- Tests behavior with minimal GPU utilization
- Reveals overhead costs and launch characteristics of different approaches
- Shows which patterns work best for tiny workloads

**Observed characteristics:**
- Launch overhead dominates computation time (~3-4ms baseline)
- Tiled/vectorize patterns show lower overhead (~3.16ms)
- Elementwise/manual show higher overhead (~4.6ms)

### 2. Medium problem size (SIZE=128, TILE=16)
```mojo
benchmark_elementwise_parameterized[128, 16]
benchmark_tiled_parameterized[128, 16]
benchmark_manual_vectorized_parameterized[128, 16]
benchmark_vectorized_parameterized[128, 16]
```

**Purpose:**
- Transition zone between overhead-dominated and computation-dominated
- Tests moderate GPU utilization
- All patterns converge to similar performance

**Observed characteristics:**
- All patterns perform similarly (~3.16ms)
- Launch overhead still significant relative to computation
- Pattern differences begin to emerge but remain subtle

### 3. Large problem size (SIZE=1,048,576, TILE=1024)
```mojo
benchmark_elementwise_parameterized[1048576, 1024]
benchmark_tiled_parameterized[1048576, 1024]
benchmark_manual_vectorized_parameterized[1048576, 1024]
benchmark_vectorized_parameterized[1048576, 1024]
```

**Purpose:**
- Tests real GPU workloads with significant computation
- Reveals true algorithmic performance differences
- Memory bandwidth becomes the primary factor

**Observed characteristics:**
- Meaningful performance scaling (11-15ms, 3× slower than small problems)
- Clear pattern differentiation emerges
- Elementwise fastest (~11.3ms), manual vectorized slowest (~15.7ms)
- Framework adapts iteration counts (70 vs 100) based on execution time

## Understanding benchmark results

### Interpreting the output

Each benchmark result shows:
```txt
| name                     | met (ms)           | iters |
| elementwise_16_4         | 3.15463684         | 100   |
```

**Metrics explained:**
- **`name`**: The benchmark identifier (pattern_size_tile format)
- **`met (ms)`**: Total execution time in milliseconds for all iterations
- **`iters`**: Number of iterations performed (typically 100)

**What to look for:**
- **Lower total time**: Better overall performance for the workload
- **Consistent timing**: Similar execution times indicate stable performance
- **Relative comparison**: Compare times between patterns for the same problem size

### Performance analysis framework

When analyzing results, consider these factors:

#### 1. **Thread utilization efficiency**
```
Small problem (SIZE=16):
Elementwise: 16 ÷ 4 = 4 threads
Tiled:       16 ÷ 4 = 4 threads
Manual:      16 ÷ (4×4) = 1 thread
Vectorize:   16 ÷ 4 = 4 threads

Medium problem (SIZE=128):
Elementwise: 128 ÷ 4 = 32 threads
Tiled:       128 ÷ 16 = 8 threads
Manual:      128 ÷ (16×4) = 2 threads
Vectorize:   128 ÷ 16 = 8 threads

Large problem (SIZE=1,048,576):
Elementwise: 1,048,576 ÷ 4 = 262,144 threads
Tiled:       1,048,576 ÷ 1024 = 1,024 threads
Manual:      1,048,576 ÷ (1024×4) = 256 threads
Vectorize:   1,048,576 ÷ 1024 = 1,024 threads
```

**Real-world observations:**
- **Small problems**: Thread count differences don't matter, overhead dominates
- **Medium problems**: Still overhead-dominated, patterns converge
- **Large problems**: Thread utilization becomes meaningful, clear performance differentiation

#### 2. **Memory access pattern impact**
```
Elementwise: Distributed access across entire array
Tiled:       Localized access within small blocks
Manual:      Large sequential chunks
Vectorize:   Automatic optimization within tiles
```

**Performance indicators:**
- **Cache hit rates**: Tiled patterns should show better cache behavior
- **Memory bandwidth**: Sequential patterns should achieve higher bandwidth
- **Latency hiding**: More threads should hide memory access latency better

#### 3. **SIMD utilization analysis**
All patterns achieve the same total SIMD operations, but with different organization:

**For SIZE=16 (SIMD_WIDTH=4):**
```
Total SIMD ops = 16 ÷ 4 = 4 SIMD operations

Distribution:
- Elementwise: 4 threads × 1 SIMD op each
- Tiled:       4 threads × 1 SIMD op each
- Manual:      1 thread × 4 SIMD ops
- Vectorize:   4 threads × 1 SIMD op each (automatic)
```

**For SIZE=128 (SIMD_WIDTH=4):**
```
Total SIMD ops = 128 ÷ 4 = 32 SIMD operations

Distribution:
- Elementwise: 32 threads × 1 SIMD op each
- Tiled:       8 threads × 4 SIMD ops each
- Manual:      2 threads × 16 SIMD ops each
- Vectorize:   8 threads × 4 SIMD ops each (automatic)
```

**For SIZE=1,048,576 (SIMD_WIDTH=4):**
```
Total SIMD ops = 1,048,576 ÷ 4 = 262,144 SIMD operations

Distribution:
- Elementwise: 262,144 threads × 1 SIMD op each
- Tiled:       1,024 threads × 256 SIMD ops each
- Manual:      256 threads × 1,024 SIMD ops each
- Vectorize:   1,024 threads × 256 SIMD ops each (automatic)
```

**Performance correlation:**
At large scale, the pattern with the highest thread count (elementwise) performs best, suggesting that GPU parallelization outweighs complex memory access optimizations for simple memory-bound operations.

## Practical performance insights

### Observed performance patterns

Based on the empirical benchmark results, here's what we actually observe:

#### **Small problems (SIZE=16)**
- **Tiled/Vectorize**: Fastest (~3.16ms) due to lower launch overhead
- **Elementwise/Manual**: Slower (~4.6ms) due to higher launch overhead
- **Launch overhead dominates**: Computation time is negligible compared to GPU kernel launch costs

#### **Medium problems (SIZE=128)**
- **All patterns converge**: Performance differences nearly disappear (~3.16ms)
- **Transitional behavior**: Still overhead-dominated but patterns begin to differentiate
- **Framework optimization**: Iteration counts automatically adjust (80-100 iterations)

#### **Large problems (SIZE=1,048,576) - Where real differences emerge**
- **Elementwise wins**: 11.34ms - High parallelism excels for memory-bound operations
- **Tiled second**: 12.04ms - Good balance of parallelism and locality
- **Vectorize third**: 13.38ms - Automatic optimization overhead becomes visible
- **Manual vectorized slowest**: 15.75ms - Complex indexing hurts simple operations

**Key insight:** For simple memory-bound operations like vector addition, **maximum parallelism (elementwise) outperforms complex memory optimizations** at scale.

### Hardware-specific considerations

Your results will vary based on:

#### **GPU architecture factors:**
- **SIMD width**: Affects optimal vectorization strategy
- **Cache size**: Influences optimal tile size
- **Core count**: Determines benefit of higher thread counts
- **Memory bandwidth**: Sets theoretical performance ceiling

#### **System factors:**
- **Memory hierarchy**: L1/L2 cache sizes affect tiling benefits
- **Thermal throttling**: May affect sustained performance
- **Concurrent workloads**: Other GPU usage impacts results

## Advanced benchmarking techniques

### Custom benchmark scenarios

You can modify the benchmark parameters to test different scenarios:

```mojo
// Test different problem sizes
benchmark_elementwise_parameterized[1024, 32]  // Large problem
benchmark_elementwise_parameterized[64, 8]     // Small problem

// Test different tile sizes
benchmark_tiled_parameterized[256, 8]   // Small tiles
benchmark_tiled_parameterized[256, 64]  // Large tiles
```

### Performance profiling integration

For deeper analysis, combine benchmarking with:
- **GPU profilers**: NVIDIA Nsight, AMD ROCProfiler
- **Memory bandwidth tools**: Check actual vs theoretical bandwidth
- **Cache analysis**: Measure hit rates and access patterns

💡 **Note**: Advanced GPU profiling techniques and tools will be covered in detail in later parts of this book series.

## Interpreting your results

### Performance ranking methodology

When comparing results:

1. **Normalize by problem size**: Calculate throughput (elements/second)
2. **Consider consistency**: Prefer patterns with low deviation
3. **Account for complexity**: Factor in development and maintenance costs
4. **Test scaling**: Verify performance across different problem sizes

### Making optimization decisions

Use benchmark results to:

#### **Choose optimal patterns based on empirical results:**
- **For simple memory-bound operations**: Elementwise pattern consistently wins at scale
- **For small/startup workloads**: Tiled or vectorize patterns have lower launch overhead
- **For development productivity**: Mojo vectorize provides good performance with automatic optimization
- **Avoid manual vectorization**: For simple operations, the complexity doesn't pay off

#### **Problem size considerations:**
- **Tiny problems (< 1K elements)**: Launch overhead dominates, choose patterns with lower overhead
- **Medium problems (1K-100K)**: Performance differences are minimal, choose for maintainability
- **Large problems (> 100K elements)**: Real algorithmic differences emerge, elementwise typically wins

#### **Performance insights from real data:**
- **Parallelism beats optimization**: For memory-bound workloads, more threads > complex memory patterns
- **Launch overhead is significant**: Small problems don't reveal true algorithmic performance
- **Framework intelligence**: Mojo's benchmark framework adapts iteration counts automatically (70-100)
- **Memory bandwidth ceiling**: All patterns hit similar performance walls at large scales

## Next steps

With benchmarking mastery:

- **Profile real applications**: Apply these patterns to actual workloads
- **Advanced GPU patterns**: Explore reductions, convolutions, and matrix operations
- **Multi-GPU scaling**: Understand distributed GPU computing patterns
- **Memory optimization**: Dive deeper into shared memory and advanced caching

💡 **Key takeaway**: Benchmarking transforms theoretical understanding into practical performance optimization. Use empirical data to make informed decisions about which patterns work best for your specific hardware and workload characteristics.

## Best practices summary

**Benchmarking best practices:**
- Always warm up GPU before measuring
- Run multiple iterations for statistical significance
- Test across different problem sizes
- Consider both peak and average performance
- Account for real-world usage patterns

**Performance optimization workflow:**
1. **Profile first**: Measure before optimizing
2. **Identify bottlenecks**: Memory vs compute bound analysis
3. **Choose patterns**: Based on workload characteristics
4. **Tune parameters**: Optimize tile sizes and thread counts
5. **Validate**: Confirm improvements with benchmarks
