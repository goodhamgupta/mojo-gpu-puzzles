from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from memory import stack_allocation
from layout import Layout, LayoutTensor
from math import iota
from sys import sizeof


# def main():
#     fn printing_kernel():
#         print("GPU Thread: [", thread_idx.x, thread_idx.y, thread_idx.z, "]")

#     var ctx = DeviceContext()
#     ctx.enqueue_function[printing_kernel](grid_dim=100, block_dim=(2, 2, 2))
#     print("This might print before the GPU has completed its work")
#     ctx.synchronize()


# def main():
#     fn block_kernel():
#         print(
#             "block: [",
#             block_idx.x,
#             block_idx.y,
#             block_idx.z,
#             "]",
#             "thread: [",
#             thread_idx.x,
#             thread_idx.y,
#             thread_idx.z,
#             "]",
#         )

#     # grid_dim defines how mayn blocks are launched
#     # block_dum defines how many threads are launched in each block
#     var ctx = DeviceContext()
#     ctx.enqueue_function[block_kernel](grid_dim=(2, 2), block_dim=2)
#     ctx.synchronize()


def main():
    alias dtype = DType.uint32
    alias blocks = 4
    alias threads = 4
    alias elements_in = blocks * threads
    var ctx = DeviceContext()
    # This is allocating global memory
    # This memory is slower compared to shared memory between threads
    # in a block
    var in_buffer = ctx.enqueue_create_buffer[dtype](elements_in)

    # 'with' block ctx manager will take care of coping data to CPU
    # and copying results back to GPU when block ends
    # map_to_host will call synchronize before writing data back to GPU
    with in_buffer.map_to_host() as host_buffer:
        iota(host_buffer.unsafe_ptr(), elements_in)
        print(host_buffer)

    alias layout = Layout.row_major(blocks, threads)
    var in_tensor = LayoutTensor[dtype, layout](in_buffer)
    alias InTensor = LayoutTensor[dtype, layout, MutableAnyOrigin]

    fn print_values_kernel(in_tensor: InTensor):
        var bid = block_idx.x
        var tid = thread_idx.x

        print("block: ", bid, "thread:", tid, "val:", in_tensor[bid, tid])

    ctx.enqueue_function[print_values_kernel](
        in_tensor, grid_dim=blocks, block_dim=threads
    )

    ctx.synchronize()

    fn multiple_kernel[multiplier: Int](in_tensor: InTensor):
        in_tensor[block_idx.x, thread_idx.x] *= multiplier

    ctx.enqueue_function[multiple_kernel[2]](
        in_tensor, grid_dim=blocks, block_dim=threads
    )

    with in_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[dtype, layout](host_buffer)
        print(host_tensor)

    # Sum reduce output

    # create buffer of size blocks(i.e 4)
    var out_buffer = ctx.enqueue_create_buffer[dtype](blocks)
    _ = out_buffer.enqueue_fill(0)

    alias out_layout = Layout.row_major(elements_in)
    alias OutTensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin]

    var out_tensor = OutTensor(out_buffer)
    # PROBLEM
    # Can't have all threads writing to same location as it will introduce
    # race conditions. Hence, we need SHARED MEMORY

    fn sum_reduce_kernel(in_tensor: InTensor, out_tensor: OutTensor):
        # This allocates memory to be shared between threads in a block prior to the
        # kernel launching. Each kernel gets a pointer to the allocated memory.
        var shared = stack_allocation[
            threads,
            Scalar[dtype],
            address_space = AddressSpace.SHARED,
        ]()

        # Place the corresponding value into shared memory
        shared[thread_idx.x] = in_tensor[block_idx.x, thread_idx.x][0]

        # Await all the threads to finish loading their values into shared memory
        barrier()

        # If this is the first thread, sum and write the result to global memory
        if thread_idx.x == 0:
            for i in range(threads):
                out_tensor[block_idx.x] += shared[i]

    ctx.enqueue_function[sum_reduce_kernel](
        in_tensor,
        out_tensor,
        grid_dim=blocks,
        block_dim=threads,
    )

    # Copy the data back to the host and print out the buffer
    with out_buffer.map_to_host() as host_buffer:
        print(host_buffer)

    fn simd_reduce_kernel(in_tensor: InTensor, out_tensor: OutTensor):
        out_tensor[block_idx.x] = in_tensor.load[4](block_idx.x, 0).reduce_add()

    ctx.enqueue_function[simd_reduce_kernel](
        in_tensor, out_tensor, grid_dim=blocks, block_dim=1
    )

    with out_buffer.map_to_host() as host_buffer:
        print(host_buffer)

    # warp -> Group of 32 threads on Nvidia GPUs
    # Allows SIMT -> Single Instruction Multiple Threads
    #  - allows multiple threads to execute same instructions on different data
    #  - with independent control flow and thread states
    # SIMD
    #  - CPU focussed
    #  - applies single instruction to multiple data WITH NO THREAD INDEPENDENCE.

    fn warp_reduce_kernel(in_tensor: InTensor, out_tensor: OutTensor):
        var value = in_tensor.load[1](block_idx.x, thread_idx.x)

        # Each thread gets the value from one thread higher, summing it
        # up as they go

        value = warp.sum(value)

        # print each reduction step
        if block_idx.x == 0:
            print("thread: ", thread_idx.x, "value: ", value)

        # thread 0 has the reduced sum of the values from all the other threads
        if thread_idx.x == 0:
            out_tensor[block_idx.x] = value

    ctx.enqueue_function[warp_reduce_kernel](
        in_tensor, out_tensor, grid_dim=blocks, block_dim=threads
    )

    with out_buffer.map_to_host() as host_buffer:
        print(host_buffer)
