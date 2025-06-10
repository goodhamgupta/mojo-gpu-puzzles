# Complete Version

Implement a kernel that computes a prefix-sum over 1D LayoutTensor `a` and stores it in 1D LayoutTensor `output`.

**Note:** _If the size of `a` is greater than the block size, we need to synchronize across multiple blocks to get the correct result._

## Configuration

- Array size: `SIZE_2 = 15` elements
- Threads per block: `TPB = 8`
- Number of blocks: 2
- Shared memory: `TPB` elements per block

Notes:

- **Multiple blocks**: When the input array is larger than one block, we need a multi-phase approach
- **Block-level sync**: Within a block, use `barrier()` to synchronize threads
- **Host-level sync**: Between blocks, use `ctx.synchronize()` at the host level
- **Auxiliary storage**: Use extra space to store block sums for cross-block communication

## Code to complete

You need to complete two separate kernel functions for the multi-block prefix sum:

1. **First kernel** (`prefix_sum_local_phase`): Computes local prefix sums within each block and stores block sums
2. **Second kernel** (`prefix_sum_block_sum_phase`): Adds previous block sums to elements in subsequent blocks

The main function will handle the necessary host-side synchronization between these kernels.

```mojo
{{#include ../../../problems/p12/p12.mojo:prefix_sum_complete}}
```
<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p12/p12.mojo" class="filename">View full file: problems/p12/p12.mojo</a>

The key to this puzzle is understanding that [barrier](https://docs.modular.com/mojo/stdlib/gpu/sync/barrier/) only synchronizes threads within a block, not across blocks. For cross-block synchronization, you need to use host-level synchronization:

```mojo
{{#include ../../../solutions/p12/p12.mojo:prefix_sum_complete_block_level_sync}}
```

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. Build on the simple prefix sum

The [Simple Version](./simple.md) shows how to implement a single-block prefix sum. You'll need to extend that approach to work across multiple blocks:

```
Simple version (single block): [0,1,2,3,4,5,6,7] → [0,1,3,6,10,15,21,28]

Complete version (two blocks):
Block 0: [0,1,2,3,4,5,6,7] → [0,1,3,6,10,15,21,28]
Block 1: [8,9,10,11,12,13,14] → [8,17,27,38,50,63,77]
```

But how do we handle the second block's values? They need to include sums from the first block!

### 2. Two-phase approach

The simple prefix sum can't synchronize across blocks, so split the work:

1. **First phase**: Each block computes its own local prefix sum (just like the simple version)
2. **Second phase**: Blocks incorporate the sums from previous blocks

Remember: `barrier()` only synchronizes threads within one block. You need host-level synchronization between phases.

### 3. Extended memory strategy

Since blocks can't directly communicate, you need somewhere to store block sums:

- Allocate extra memory at the end of your output buffer
- Last thread in each block stores its final sum in this extra space
- Subsequent blocks can read these sums and add them to their elements

### 4. Key implementation insights

- **Different layouts**: Input and output may have different shapes
- **Boundary handling**: Always check `global_i < size` for array bounds
- **Thread role specialization**: Only specific threads (e.g., last thread) should store block sums
- **Two kernel synchronization**: Use `ctx.synchronize()` between kernel launches

### 5. Debugging Strategy

If you encounter issues, try visualizing the intermediate state after the first phase:
```
After first phase: [0,1,3,6,10,15,21,28, 8,17,27,38,50,63,77, ???,???]
```

Where `???` should contain your block sums that will be used in the second phase.

</div>
</details>

### Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p12 --complete
```

  </div>
  <div class="tab-content">

```bash
pixi run p12 --complete
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:
```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0, 91.0, 105.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p12/p12.mojo:prefix_sum_complete_solution}}
```

<div class="solution-explanation">

This solution implements a multi-block prefix sum using a two-kernel approach to handle an array that spans multiple thread blocks. Let's break down each aspect in detail:

## The challenge of cross-block communication

The fundamental limitation in GPU programming is that threads can only synchronize within a block using `barrier()`. When data spans multiple blocks, we face the challenge: **How do we ensure blocks can communicate their partial results to other blocks?**

### Memory layout visualization

For our test case with `SIZE_2 = 15` and `TPB = 8`:

```
Input array:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

Block 0 processes: [0, 1, 2, 3, 4, 5, 6, 7]
Block 1 processes: [8, 9, 10, 11, 12, 13, 14, (padding)]
```

We extend the output buffer to include space for block sums:

```
Extended buffer: [data values (15 elements)] + [block sums (2 elements)]
                 [0...14] + [block0_sum, block1_sum]
```

The size of this extended buffer is: `EXTENDED_SIZE = SIZE_2 + num_blocks = 15 + 2 = 17`

## Phase 1 kernel: Local prefix sums

### Step-by-step execution for Block 0

1. **Load values into shared memory**:
   ```
   shared = [0, 1, 2, 3, 4, 5, 6, 7]
   ```

2. **Iterations of parallel reduction** (\\(\log_2(TPB) = 3\\) iterations):

   **Iteration 1** (offset=1):
   ```
   shared[0] = 0              (unchanged)
   shared[1] = 1 + 0 = 1
   shared[2] = 2 + 1 = 3
   shared[3] = 3 + 2 = 5
   shared[4] = 4 + 3 = 7
   shared[5] = 5 + 4 = 9
   shared[6] = 6 + 5 = 11
   shared[7] = 7 + 6 = 13
   ```
   After barrier: `shared = [0, 1, 3, 5, 7, 9, 11, 13]`

   **Iteration 2** (offset=2):
   ```
   shared[0] = 0              (unchanged)
   shared[1] = 1              (unchanged)
   shared[2] = 3 + 0 = 3      (unchanged)
   shared[3] = 5 + 1 = 6
   shared[4] = 7 + 3 = 10
   shared[5] = 9 + 5 = 14
   shared[6] = 11 + 7 = 18
   shared[7] = 13 + 9 = 22
   ```
   After barrier: `shared = [0, 1, 3, 6, 10, 14, 18, 22]`

   **Iteration 3** (offset=4):
   ```
   shared[0] = 0              (unchanged)
   shared[1] = 1              (unchanged)
   shared[2] = 3              (unchanged)
   shared[3] = 6              (unchanged)
   shared[4] = 10 + 0 = 10    (unchanged)
   shared[5] = 14 + 1 = 15
   shared[6] = 18 + 3 = 21
   shared[7] = 22 + 6 = 28
   ```
   After barrier: `shared = [0, 1, 3, 6, 10, 15, 21, 28]`

3. **Write local results back to global memory**:
   ```
   output[0...7] = [0, 1, 3, 6, 10, 15, 21, 28]
   ```

4. **Store block sum in auxiliary space** (only last thread):
   ```
   output[15] = 28  // at position size + block_idx.x = 15 + 0
   ```

### Step-by-step execution for Block 1

1. **Load values into shared memory**:
   ```
   shared = [8, 9, 10, 11, 12, 13, 14, 0] // Last value padded with 0
   ```

2. **Iterations of parallel reduction** (\\(\log_2(TPB) = 3\\) iterations):

   With similar iterations as Block 0, after all three iterations:
   ```
   shared = [8, 17, 27, 38, 50, 63, 77, 77]
   ```

3. **Write local results back to global memory**:
   ```
   output[8...14] = [8, 17, 27, 38, 50, 63, 77]
   ```

4. **Store block sum in auxiliary space** (only last thread):
   ```
   output[16] = 77  // at position size + block_idx.x = 15 + 1
   ```

After Phase 1, the output buffer contains:
```
[0, 1, 3, 6, 10, 15, 21, 28, 8, 17, 27, 38, 50, 63, 77, 28, 77]
                                                        ^   ^
                                                Block sums stored here
```

## Host-side synchronization: The critical step

Between phases 1 and 2, we call:
```mojo
ctx.synchronize()
```

This is the most crucial part of the algorithm! Without this synchronization, the second kernel might start before the first one completes, leading to race conditions and incorrect results. This is a fundamental difference from single-block algorithms where `barrier()` would be sufficient.

## Phase 2 kernel: Block sum addition

1. **Block 0**: No changes needed (it's already correct).

2. **Block 1**: Each thread adds Block 0's sum to its element:
   ```
   prev_block_sum = output[size + block_idx.x - 1] = output[15] = 28
   output[global_i] += prev_block_sum
   ```

   Block 1 values are transformed:
   ```
   Before: [8, 17, 27, 38, 50, 63, 77]
   After:  [36, 45, 55, 66, 78, 91, 105]
   ```

## Performance and optimization considerations

1. **Work efficiency**: This implementation has \\(O(n \log n)\\) work complexity, while the sequential algorithm is \\(O(n)\\). This is a classic space-time tradeoff in parallel algorithms.

2. **Memory overhead**: The extra space for block sums is minimal (just one element per block).

This two-kernel approach is a fundamental pattern in GPU programming for algorithms that require cross-block communication. The same strategy can be applied to other parallel algorithms like radix sort, histogram calculation, and reduction operations.
</div>
</details>
