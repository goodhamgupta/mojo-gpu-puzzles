# Summary

# Getting Started
- [🔥 Introduction](./introduction.md)
- [🧭 Puzzles Usage Guide](./howto.md)

# Part I: GPU Fundamentals
- [Puzzle 1: Map](./puzzle_01/puzzle_01.md)
  - [🔰 Raw Memory Approach](./puzzle_01/raw.md)
  - [💡 Preview: Modern Approach with LayoutTensor](./puzzle_01/layout_tensor_preview.md)
- [Puzzle 2: Zip](./puzzle_02/puzzle_02.md)
- [Puzzle 3: Guards](./puzzle_03/puzzle_03.md)
- [Puzzle 4: 2D Map](./puzzle_04/puzzle_04.md)
  - [🔰 Raw Memory Approach](./puzzle_04/raw.md)
  - [📚 Learn about LayoutTensor](./puzzle_04/introduction_layout_tensor.md)
  - [🚀 Modern 2D Operations](./puzzle_04/layout_tensor.md)
- [Puzzle 5: Broadcast](./puzzle_05/puzzle_05.md)
  - [🔰 Raw Memory Approach](./puzzle_05/raw.md)
  - [📐 LayoutTensor Version](./puzzle_05/layout_tensor.md)
- [Puzzle 6: Blocks](./puzzle_06/puzzle_06.md)
- [Puzzle 7: 2D Blocks](./puzzle_07/puzzle_07.md)
  - [🔰 Raw Memory Approach](./puzzle_07/raw.md)
  - [📐 LayoutTensor Version](./puzzle_07/layout_tensor.md)
- [Puzzle 8: Shared Memory](./puzzle_08/puzzle_08.md)
  - [🔰 Raw Memory Approach](./puzzle_08/raw.md)
  - [📐 LayoutTensor Version](./puzzle_08/layout_tensor.md)

# Part II: 🧮 GPU Algorithms
- [Puzzle 9: Pooling](./puzzle_09/puzzle_09.md)
  - [🔰 Raw Memory Approach](./puzzle_09/raw.md)
  - [📐 LayoutTensor Version](./puzzle_09/layout_tensor.md)
- [Puzzle 10: Dot Product](./puzzle_10/puzzle_10.md)
  - [🔰 Raw Memory Approach](./puzzle_10/raw.md)
  - [📐 LayoutTensor Version](./puzzle_10/layout_tensor.md)
- [Puzzle 11: 1D Convolution](./puzzle_11/puzzle_11.md)
  - [🔰 Simple Version](./puzzle_11/simple.md)
  - [⭐ Block Boundary Version](./puzzle_11/block_boundary.md)
- [Puzzle 12: Prefix Sum](./puzzle_12/puzzle_12.md)
  - [🔰 Simple Version](./puzzle_12/simple.md)
  - [⭐ Complete Version](./puzzle_12/complete.md)
- [Puzzle 13: Axis Sum](./puzzle_13/puzzle_13.md)
- [Puzzle 14: Matrix Multiplication (MatMul)](./puzzle_14/puzzle_14.md)
    - [🔰 Naïve Version with Global Memory](./puzzle_14/naïve.md)
    - [📚 Learn about Roofline Model](./puzzle_14/roofline.md)
    - [🤝 Shared Memory Version](./puzzle_14/shared_memory.md)
    - [📐 Tiled Version](./puzzle_14/tiled.md)

# Part III: 🐍 Interfacing with Python via MAX Graph Custom Ops
- [Puzzle 15: 1D Convolution Op](./puzzle_15/puzzle_15.md)
- [Puzzle 16: Softmax Op](./puzzle_16/puzzle_16.md)
- [Puzzle 17: Attention Op](./puzzle_17/puzzle_17.md)
- [🎯 Bonus Challenges](./bonuses/part3.md)

# Part IV: 🔥 PyTorch Custom Ops Integration
- [Puzzle 18: 1D Convolution Op](./puzzle_18/puzzle_18.md)
- [Puzzle 19: Embedding Op](./puzzle_19/puzzle_19.md)
  - [🔰 Coaleasced vs non-Coaleasced Kernel](./puzzle_19/simple_embedding_kernel.md)
  - [📊 Performance Comparison](./puzzle_19/performance.md)
- [Puzzle 20: Kernel Fusion and Custom Backward Pass](./puzzle_20/puzzle_20.md)
  - [⚛️ Fused vs Unfused Kernels](./puzzle_20/forward_pass.md)
  - [⛓️ Autograd Integration & Backward Pass](./puzzle_20/backward_pass.md)

# Part V: 🌊 Mojo Functional Patterns and Benchmarking
- [Puzzle 21: GPU Functional Programming Patterns](./puzzle_21/puzzle_21.md)
  - [elementwise - Basic GPU Functional Operations](./puzzle_21/elementwise.md)
  - [tile - Memory-Efficient Tiled Processing](./puzzle_21/tile.md)
  - [Vectorization - SIMD Control](./puzzle_21/vectorize.md)
  - [🧠 GPU Threading vs SIMD Concepts](./puzzle_21/gpu-thread-vs-simd.md)
  - [📊 Benchmarking in Mojo](./puzzle_21/benchmarking.md)

# Part VI: ⚡ Warp-Level Programming
- [Puzzle 22: Warp Fundamentals](./puzzle_22/puzzle_22.md)
  - [🧠 Warp lanes & SIMT execution](./puzzle_22/warp_simt.md)
  - [🔰 warp.sum() Essentials](./puzzle_22/warp_sum.md)
  - [📊 When to Use Warp Programming](./puzzle_22/warp_extra.md)
- [Puzzle 23: Warp Communication](./puzzle_23/puzzle_23.md)
  - [⬇️ warp.shuffle_down()](./puzzle_23/warp_shuffle_down.md)
  - [📢 warp.broadcast()](./puzzle_23/warp_broadcast.md)
- [Puzzle 24: Advanced Warp Patterns](./puzzle_24/puzzle_24.md)
  - [🦋 warp.shuffle_xor() Butterfly Networks](./puzzle_24/warp_shuffle_xor.md)
  - [🔢 warp.prefix_sum() Scan Operations](./puzzle_24/warp_prefix_sum.md)

# Part VII: Advanced Memory Operations
- [Puzzle 25: Memory Coalescing]()
  - [📚 Understanding Coalesced Access]()
  - [Optimized Access Patterns]()
  - [🔧 Troubleshooting Memory Issues]()
- [Puzzle 26: Async Memory Operations]()
- [Puzzle 27: Memory Fences & Atomics]()
- [Puzzle 28: Prefetching & Caching]()

# Part VIII: 📊 Performance Analysis & Optimization
- [Puzzle 29: GPU Profiling Basics]()
- [Puzzle 30: Occupancy Optimization]()
- [Puzzle 31: Bank Conflicts]()
  - [📚 Understanding Shared Memory Banks]()
  - [Conflict-Free Patterns]()

# Part IX: 🚀 Advanced GPU Features
- [Puzzle 32: Tensor Core Operations]()
- [Puzzle 33: Random Number Generation]()
- [Puzzle 34: Advanced Synchronization]()

# Part X: 🌐 Multi-GPU & Advanced Applications
- [Puzzle 35: Multi-Stream Programming]()
- [Puzzle 36: Multi-GPU Basics]()
- [Puzzle 37: End-to-End Optimization Case Study]()
- [🎯 Advanced Bonus Challenges]()
