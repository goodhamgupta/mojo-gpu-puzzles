# 🧠 GPU Threading vs SIMD - Understanding the Execution Hierarchy

## Overview

After exploring **elementwise**, **tiled**, and **vectorization** patterns, you've seen different ways to organize GPU computation. This section clarifies the fundamental relationship between **GPU threads** and **SIMD operations** - two distinct but complementary levels of parallelism that work together for optimal performance.

Understanding this relationship is crucial for making informed decisions about which patterns to use for different workloads.

**Key insight:** _GPU threads provide the parallelism structure, while SIMD operations provide the vectorization within each thread._

## Core concepts

### GPU threading hierarchy

GPU execution follows a well-defined hierarchy that abstracts hardware complexity:

```
GPU Device
├── Grid (your entire problem)
│   ├── Block 1 (group of threads, shared memory)
│   │   ├── Warp 1 (32 threads, lockstep execution)
│   │   │   ├── Thread 1 → Your Mojo function
│   │   │   ├── Thread 2 → Your Mojo function
│   │   │   └── ... (32 threads total)
│   │   └── Warp 2 (32 threads)
│   └── Block 2 (independent group)
```

**What Mojo abstracts for you:**
- **Grid/Block configuration**: Automatically calculated based on problem size
- **Warp management**: Hardware handles 32-thread groups transparently
- **Thread scheduling**: GPU scheduler manages execution automatically
- **Memory hierarchy**: Optimal access patterns built into functional operations

### SIMD within GPU threads

Each GPU thread can process multiple data elements simultaneously using **SIMD (Single Instruction, Multiple Data)** operations:

```mojo
// Within one GPU thread:
a_simd = a.load[simd_width](idx, 0)    // Load 4 floats simultaneously
b_simd = b.load[simd_width](idx, 0)    // Load 4 floats simultaneously
result = a_simd + b_simd               // Add 4 pairs simultaneously
out.store[simd_width](idx, 0, result)  // Store 4 results simultaneously
```

**SIMD characteristics:**
- **Width**: Determined by GPU architecture and data type (typically 4, 8, or 16)
- **Efficiency**: Single instruction operates on multiple data elements
- **Memory**: Vectorized loads/stores improve bandwidth utilization
- **Hardware**: Maps directly to GPU vector processing units

## Pattern comparison and thread-to-work mapping

Let's analyze how each pattern maps threads to work using our `SIZE = 1024` example:

### 1. Elementwise pattern
```mojo
elementwise[add_function, simd_width, target="gpu"](size, ctx)
```

**Thread organization:**
- **Thread count**: `1024 ÷ 4 = 256` threads (for SIMD_WIDTH=4)
- **Work per thread**: Exactly `simd_width` elements (4 elements)
- **Memory pattern**: Each thread accesses 4 consecutive elements
- **SIMD usage**: One SIMD operation per thread

**Execution visualization:**
```
Thread 0: processes elements [0, 1, 2, 3]     → 1 SIMD operation
Thread 1: processes elements [4, 5, 6, 7]     → 1 SIMD operation
Thread 2: processes elements [8, 9, 10, 11]   → 1 SIMD operation
...
Thread 255: processes elements [1020, 1021, 1022, 1023] → 1 SIMD operation
Total: 256 threads × 1 SIMD op = 256 SIMD operations
```

### 2. Tiled pattern
```mojo
elementwise[process_tiles, 1, target="gpu"](num_tiles, ctx)
```

**Thread organization:**
- **Thread count**: `1024 ÷ 32 = 32` threads (for TILE_SIZE=32)
- **Work per thread**: 32 elements processed sequentially
- **Memory pattern**: Each thread accesses a contiguous 32-element block
- **SIMD usage**: `32 ÷ 4 = 8` SIMD operations per thread

**Execution visualization:**
```
Thread 0: processes elements [0:32]      → 8 SIMD operations (sequential)
Thread 1: processes elements [32:64]     → 8 SIMD operations (sequential)
Thread 2: processes elements [64:96]     → 8 SIMD operations (sequential)
...
Thread 31: processes elements [992:1024] → 8 SIMD operations (sequential)
Total: 32 threads × 8 SIMD ops = 256 SIMD operations
```

### 3. Manual vectorization pattern
```mojo
elementwise[manual_vectorized, 1, target="gpu"](num_chunks, ctx)
```

**Thread organization:**
- **Thread count**: `1024 ÷ 128 = 8` threads (for chunk_size=128)
- **Work per thread**: 128 elements in 32 SIMD groups
- **Memory pattern**: Large chunks with stride-4 access within chunks
- **SIMD usage**: `128 ÷ 4 = 32` SIMD operations per thread

**Execution visualization:**
```
Thread 0: processes elements [0:128]     → 32 SIMD operations
Thread 1: processes elements [128:256]   → 32 SIMD operations
Thread 2: processes elements [256:384]   → 32 SIMD operations
...
Thread 7: processes elements [896:1024]  → 32 SIMD operations
Total: 8 threads × 32 SIMD ops = 256 SIMD operations
```

### 4. Mojo vectorize pattern
```mojo
elementwise[vectorize_tiles, 1, target="gpu"](num_tiles, ctx)
vectorize[nested_function, simd_width](tile_size)
```

**Thread organization:**
- **Thread count**: `1024 ÷ 32 = 32` threads (for `TILE_SIZE=32`)
- **Work per thread**: 32 elements with automatic vectorization
- **Memory pattern**: Contiguous tiles with automatic SIMD chunking
- **SIMD usage**: `32 ÷ 4 = 8` SIMD operations per thread (automatic)

**Execution visualization:**
```
Thread 0: processes tile [0:32]    → 8 automatic SIMD operations
Thread 1: processes tile [32:64]   → 8 automatic SIMD operations
Thread 2: processes tile [64:96]   → 8 automatic SIMD operations
...
Thread 31: processes tile [992:1024] → 8 automatic SIMD operations
Total: 32 threads × 8 SIMD ops = 256 SIMD operations
```

## Key insights and trade-offs

### 1. **Total work remains constant**
All patterns achieve the same total computational work:
- **256 SIMD operations** across all approaches
- **1024 elements processed** in all cases
- **Same arithmetic intensity** for this simple operation

### 2. **Thread count vs work per thread trade-off**
```
Elementwise:  256 threads  × 1 SIMD op   = High parallelism, minimal work
Tiled:        32 threads   × 8 SIMD ops  = Moderate parallelism, moderate work
Manual:       8 threads    × 32 SIMD ops = Low parallelism, substantial work
Mojo vectorize: 32 threads × 8 SIMD ops  = Moderate parallelism, automatic work
```

### 3. **Memory access pattern implications**

**Elementwise**:
- **Access pattern**: Distributed across entire array
- **Cache behavior**: Poor spatial locality between threads
- **Bandwidth**: Excellent coalescing within SIMD operations

**Tiled/Vectorize**:
- **Access pattern**: Localized to small contiguous blocks
- **Cache behavior**: Excellent spatial locality within threads
- **Bandwidth**: Good coalescing + excellent cache reuse

**Manual vectorization**:
- **Access pattern**: Large contiguous blocks
- **Cache behavior**: May exceed cache capacity, but perfect sequentiality
- **Bandwidth**: Maximum theoretical bandwidth utilization

### 4. **Hardware utilization considerations**

**GPU occupancy factors:**
- **Thread count**: More threads = better latency hiding capability
- **Memory per thread**: More work per thread = better cache utilization
- **Register pressure**: Complex operations may limit threads per SM
- **Shared memory usage**: Advanced patterns may require shared memory

**Warp efficiency:**
- All patterns launch threads in warp-aligned groups (multiples of 32)
- GPU scheduler handles warp-level execution automatically
- Memory stalls in one warp allow other warps to execute

## Choosing the right pattern

### **Use elementwise when:**
- **Simple operations** with minimal arithmetic per element
- **Maximum parallelism** is needed for latency hiding
- **Regular data access** patterns with good coalescing
- **Scalability** across different problem sizes is important

### **Use tiled when:**
- **Cache-sensitive** operations that benefit from data reuse
- **Moderate complexity** operations within each element
- **Memory bandwidth** is not the primary bottleneck
- **Sequential access** patterns provide better performance

### **Use manual vectorization when:**
- **Expert-level control** over memory access patterns is needed
- **Complex indexing** or non-standard access patterns are required
- **Maximum memory bandwidth** utilization is critical
- **Hardware-specific optimization** is worth the complexity

### **Use Mojo vectorize when:**
- **Development productivity** and safety are priorities
- **Automatic optimization** is preferred over manual tuning
- **Bounds checking** and edge case handling add value
- **Portability** across different hardware is important

## Performance mental model

**Think of it this way:**
- **GPU threads** provide the **parallel structure** - how many independent execution units
- **SIMD operations** provide the **vectorization** - how efficiently each unit processes data
- **Memory patterns** determine the **bandwidth utilization** - how effectively you use the memory subsystem

**Optimal performance requires:**
1. **Sufficient parallelism**: Enough threads to hide memory latency
2. **Efficient vectorization**: Maximize SIMD utilization within threads
3. **Optimal memory patterns**: Achieve high bandwidth with good cache behavior

## Advanced considerations

### **Hardware mapping reality**
While we think in terms of "GPU threads," the hardware reality is more complex:
- **Warps**: 32 threads execute in lockstep
- **Streaming Multiprocessors (SMs)**: Multiple warps execute concurrently
- **SIMD units**: Vector processing units within each SM
- **Memory hierarchy**: L1/L2 caches, shared memory, global memory

**Mojo abstracts this complexity** while still allowing you to reason about performance through the thread/SIMD mental model.

### **Scaling considerations**
```
Small problems (< 1K elements):  Elementwise often optimal
Medium problems (1K - 1M):       Tiled patterns often best
Large problems (> 1M):           Manual vectorization may excel
```

The optimal choice depends on:
- **Problem size** relative to GPU capabilities
- **Arithmetic complexity** of your operations
- **Memory access patterns** of your algorithm
- **Development time** vs performance requirements

## Next steps

With a solid understanding of GPU threading vs SIMD concepts:

- **[📊 Benchmarking](./benchmarking.md)**: Measure and compare actual performance

💡 **Key takeaway**: GPU threads and SIMD operations work together as complementary levels of parallelism. Understanding their relationship allows you to choose the right pattern for your specific performance requirements and constraints.
