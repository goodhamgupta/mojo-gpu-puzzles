# Puzzle 1: Map

## Overview

GPU programming is all about parallelism. In this puzzle, each thread will process a single element of the input array independently.
Implement a kernel that adds 10 to each position of vector `a` and stores it in vector `output`.

**Note:** _You have 1 thread per position._

![Map](./media/videos/720p30/puzzle_01_viz.gif)

## Key concepts
- Basic GPU kernel structure
- One-to-one thread to data mapping
- Memory access patterns
- Array operations on GPU

For each position \\(i\\):
\\[\Large output[i] = a[i] + 10\\]

## What we cover

### [🔰 Raw Memory Approach](./raw.md)
Start with direct memory manipulation to understand GPU fundamentals.

### [💡 Preview: Modern Approach with LayoutTensor](./layout_tensor_preview.md)
See how LayoutTensor simplifies GPU programming with safer, cleaner code.

💡 **Tip**: Understanding both approaches helps you better appreciate modern GPU programming patterns.
