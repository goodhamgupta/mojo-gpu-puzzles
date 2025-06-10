# Tiled Matrix Multiplication

## Overview

Implement a kernel that multiplies square matrices \\(A\\) and \\(B\\) using tiled matrix multiplication with LayoutTensor. This approach handles large matrices by processing them in smaller chunks (tiles).

## Key concepts

- Matrix tiling with LayoutTensor for efficient computation
- Multi-block coordination with proper layouts
- Efficient shared memory usage through TensorBuilder
- Boundary handling for tiles with LayoutTensor indexing

## Configuration

- Matrix size: \\(\\text{SIZE\_TILED} = 8\\)
- Threads per block: \\(\\text{TPB} \times \\text{TPB} = 3 \times 3\\)
- Grid dimensions: \\(3 \times 3\\) blocks
- Shared memory: Two \\(\\text{TPB} \times \\text{TPB}\\) LayoutTensors per block

Layout configuration:
- Input A: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- Input B: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- Output: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- Shared Memory: Two `TPB × TPB` LayoutTensors using TensorBuilder

## Tiling strategy

### Block organization
```txt
Grid Layout (3×3):           Thread Layout per Block (3×3):
[B00][B01][B02]               [T00 T01 T02]
[B10][B11][B12]               [T10 T11 T12]
[B20][B21][B22]               [T20 T21 T22]

Each block processes a tile using LayoutTensor indexing
```

### Tile processing steps

1. Calculate global and local indices for thread position
2. Allocate shared memory for A and B tiles
3. For each tile:
   - Reset shared memory
   - Load tile from matrix A and B
   - Compute partial products
   - Accumulate results in registers
4. Write final accumulated result

### Memory access pattern
```txt
Matrix A (8×8)                 Matrix B (8×8)               Matrix C (8×8)
+---+---+---+                  +---+---+---+                +---+---+---+
|T00|T01|T02| ...              |T00|T01|T02| ...            |T00|T01|T02| ...
+---+---+---+                  +---+---+---+                +---+---+---+
|T10|T11|T12|                  |T10|T11|T12|                |T10|T11|T12|
+---+---+---+                  +---+---+---+                +---+---+---+
|T20|T21|T22|                  |T20|T21|T22|                |T20|T21|T22|
+---+---+---+                  +---+---+---+                +---+---+---+
  ...                            ...                          ...

Tile Processing (for computing C[T11]):
1. Load tiles from A and B:
   +---+      +---+
   |A11| ×    |B11|     For each phase k:
   +---+      +---+     C[T11] += A[row, k] × B[k, col]

2. Tile movement:
   Phase 1     Phase 2     Phase 3
   A: [T10]    A: [T11]    A: [T12]
   B: [T01]    B: [T11]    B: [T21]

3. Each thread (i,j) in tile computes:
   C[i,j] = Σ (A[i,k] × B[k,j]) for k in tile width

Synchronization required:
* After loading tiles to shared memory
* After computing each phase
```

## Code to complete

```mojo
{{#include ../../../problems/p14/p14.mojo:matmul_tiled}}
```
<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p14/p14.mojo" class="filename">View full file: problems/p14/p14.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Calculate global thread positions from block and thread indices correctly
2. Clear shared memory before loading new tiles
3. Load tiles with proper bounds checking
4. Accumulate results across tiles with proper synchronization
</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p14 --tiled
```

  </div>
  <div class="tab-content">

```bash
pixi run p14 --tiled
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:
```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([2240.0, 2296.0, 2352.0, 2408.0, 2464.0, 2520.0, 2576.0, 2632.0, 5824.0, 6008.0, 6192.0, 6376.0, 6560.0, 6744.0, 6928.0, 7112.0, 9408.0, 9720.0, 10032.0, 10344.0, 10656.0, 10968.0, 11280.0, 11592.0, 12992.0, 13432.0, 13872.0, 14312.0, 14752.0, 15192.0, 15632.0, 16072.0, 16576.0, 17144.0, 17712.0, 18280.0, 18848.0, 19416.0, 19984.0, 20552.0, 20160.0, 20856.0, 21552.0, 22248.0, 22944.0, 23640.0, 24336.0, 25032.0, 23744.0, 24568.0, 25392.0, 26216.0, 27040.0, 27864.0, 28688.0, 29512.0, 27328.0, 28280.0, 29232.0, 30184.0, 31136.0, 32088.0, 33040.0, 33992.0])
```

## Solution: Manual tiling

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p14/p14.mojo:matmul_tiled_solution}}
```

<div class="solution-explanation">

The tiled matrix multiplication implementation demonstrates efficient handling of large matrices \\((8 \times 8)\\) using small tiles \\((3 \times 3)\\). Here's how it works:

1. **Thread indexing setup**
   - Global position calculation:
     ```txt
     tiled_row = block_idx.y * TPB + thread_idx.y
     tiled_col = block_idx.x * TPB + thread_idx.x
     ```
   - Local position in tile:
     ```txt
     local_row = thread_idx.y
     local_col = thread_idx.x
     ```

2. **Shared memory allocation**
   ```txt
   Input matrices (8×8):
   A = [0  1  2  3  4  5  6  7 ]    B = [0  2  4  6  8  10 12 14]
       [8  9  10 11 12 13 14 15]        [16 18 20 22 24 26 28 30]
       [16 17 18 19 20 21 22 23]        [32 34 36 38 40 42 44 46]
       [24 25 26 27 28 29 30 31]        [48 50 52 54 56 58 60 62]
       [32 33 34 35 36 37 38 39]        [64 66 68 70 72 74 76 78]
       [40 41 42 43 44 45 46 47]        [80 82 84 86 88 90 92 94]
       [48 49 50 51 52 53 54 55]        [96 98 100 102 104 106 108 110]
       [56 57 58 59 60 61 62 63]        [112 114 116 118 120 122 124 126]

   Shared memory per block (3×3):
   a_shared[TPB, TPB]  b_shared[TPB, TPB]
   ```

3. **Tile processing loop**
   ```txt
   Number of tiles = (8 + 3 - 1) // 3 = 3 tiles

   For each tile:
   1. Reset shared memory
   2. Load tile from A and B
   3. Compute partial products
   4. Accumulate in register
   ```

4. **Memory loading pattern**
   - Loading A tile:
     ```txt
     if tiled_row < size and (tile * TPB + local_col) < size:
         a_shared[local_row, local_col] = a[tiled_row, tile * TPB + local_col]
     ```
   - Loading B tile:
     ```txt
     if (tile * TPB + local_row) < size and tiled_col < size:
         b_shared[local_row, local_col] = b[tile * TPB + local_row, tiled_col]
     ```

5. **Computation within tile**
   ```txt
   For k in range(min(TPB, size - tile * TPB)):
       acc += a_shared[local_row, k] * b_shared[k, local_col]
   ```
   - Maximizes memory coalescing:
     ```txt
     Coalesced Access (Good):          Non-Coalesced Access (Bad):
     Thread0: [M0][M1][M2][M3]         Thread0: [M0][ ][ ][ ]
     Thread1: [M4][M5][M6][M7]    vs   Thread1: [ ][M1][ ][ ]
     Thread2: [M8][M9][MA][MB]         Thread2: [ ][ ][M2][ ]
     Thread3: [MC][MD][ME][MF]         Thread3: [ ][ ][ ][M3]
     ↓                                 ↓
     1 memory transaction              4 memory transactions
     ```
     When threads access consecutive memory locations (left), the GPU can combine multiple reads into a single transaction.
     When threads access scattered locations (right), each access requires a separate transaction, reducing performance.


6. **Synchronization points**
   ```txt
   barrier() after:
   1. Shared memory reset
   2. Tile loading
   3. Tile computation
   ```

Key performance features:
- Processes 8×8 matrix using 3×3 tiles
- Uses shared memory for fast tile access
- Minimizes global memory transactions
- Handles matrix boundaries correctly
- Maintains coalesced memory access

2. **Boundary handling**:
   ```mojo
   if row < size and col < size:
       output[row, col] = acc
   ```
   - Prevents out-of-bounds access
   - Handles matrix edges
   - Safe result writing

### Key optimizations

1. **Layout optimization**:
   - Row-major layout for all tensors
   - Efficient 2D indexing

2. **Memory access**:
   - Coalesced global memory loads
   - Efficient shared memory usage

3. **Computation**:
   - Register-based accumulation i.e. `var acc: output.element_type = 0`
   - Compile-time loop unrolling via `@parameter`

This implementation achieves high performance through:
- Efficient use of LayoutTensor for memory access
- Optimal tiling strategy
- Proper thread synchronization
- Careful boundary handling
</div>
</details>

## Solution: Idiomatic LayoutTensor tiling

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p14/p14.mojo:matmul_idiomatic_tiled_solution}}
```

<div class="solution-explanation">

The idiomatic tiled matrix multiplication leverages Mojo's LayoutTensor API and asynchronous memory operations for a cleaner implementation:

1. **LayoutTensor tile API**
   ```mojo
   out_tile = output.tile[TPB, TPB](block_idx.y, block_idx.x)
   a_tile = a.tile[TPB, TPB](block_idx.y, idx)
   b_tile = b.tile[TPB, TPB](idx, block_idx.x)
   ```
   This directly expresses "get the tile at position (block_idx.y, block_idx.x)" without manual coordinate calculation. See the [documentation](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/#tile) for more details.

2. **Asynchronous memory operations**
   ```mojo
   copy_dram_to_sram_async[thread_layout=load_a_layout](a_shared, a_tile)
   copy_dram_to_sram_async[thread_layout=load_b_layout](b_shared, b_tile)
   async_copy_wait_all()
   ```
   These operations:
   - Launch asynchronous memory transfers that may overlap with computation via [copy_dram_to_sram_async](https://docs.modular.com/mojo/kernels/layout/layout_tensor/copy_dram_to_sram_async/)
   - Use specialized thread layouts for optimal memory access patterns
   - Eliminate the need for manual memory initialization

3. **Specialized compile-time load layouts**
   ```mojo
   alias load_a_layout = Layout.row_major(1, TPB)
   alias load_b_layout = Layout.row_major(TPB, 1)
   ```
   These layouts optimize how threads cooperate during memory transfers:
   - `load_a_layout`: Each thread loads a slice of a row (coalesced access)
   - `load_b_layout`: Each thread loads a slice of a column (transposed access)

4. **Efficient thread synchronization**
   ```mojo
   // Wait for async operations to complete
   async_copy_wait_all()
   // Ensure all threads can see the shared memory contents
   barrier()
   ```
   The barriers ensure proper synchronization:
   - After memory transfers complete
   - After computation for each tile

5. **Proper boundary handling**
   ```mojo
   if block_idx.y * TPB + local_row < size and block_idx.x * TPB + local_col < size:
       out_tile[local_row, local_col] = acc
   ```
   This critical check prevents out-of-bounds writes for blocks at the matrix boundaries.

6. **Tile processing loop**
   ```mojo
   for idx in range((size + TPB - 1) // TPB):
      // Process one tile
   ```
   Uses ceiling division to handle matrices whose dimensions aren't perfect multiples of the tile size.

### Performance considerations

The idiomatic implementation maintains the performance benefits of tiling while providing cleaner abstractions:

1. **Memory locality**: Exploits spatial and temporal locality through tiling
2. **Coalesced access**: Specialized load layouts ensure coalesced memory access patterns
3. **Compute-memory overlap**: Potential overlap through asynchronous memory operations
4. **Shared memory efficiency**: No redundant initialization of shared memory
5. **Register pressure**: Uses accumulation registers for optimal compute throughput

This implementation shows how high-level abstractions can express complex GPU algorithms without sacrificing performance. It's a prime example of Mojo's philosophy: combining high-level expressiveness with low-level performance control.

### Key differences from manual tiling

| Feature | Manual Tiling | Idiomatic Tiling |
|---------|--------------|------------------|
| Memory access | Direct indexing with bounds checks | LayoutTensor tile API |
| Tile loading | Explicit element-by-element copying | Asynchronous bulk transfers |
| Shared memory | Manual initialization (zeroing) | Managed by copy functions |
| Code complexity | More verbose with explicit indexing | More concise with higher-level APIs |
| Bounds checking | Multiple checks during loading and computing | Single check at final write |

The idiomatic approach is not just cleaner but also potentially more performant due to the use of specialized memory layouts and asynchronous operations.
</div>
</details>
