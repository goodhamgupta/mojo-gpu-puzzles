# Instructions

# Create a host buffer for the input of DType Float32, with 32 elements, and initialize the numbers ordered sequentially. Copy the host buffer to the device.
# Create a in_tensor that wraps the host buffer, with the dimensions (8, 4)
# Create an host and device buffer for the output of DType Float32, with 8 elements, don't forget to zero the values with enqueue_memset().
# Launch a GPU kernel with 8 blocks and 4 threads that reduce sums the values, using your preferred method to write to the output buffer.
# Copy the device buffer to the host buffer, and print it out on the CPU.

from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from memory import stack_allocation
from layout import Layout, LayoutTensor
from math import iota
from sys import sizeof


def main():
    # Init
    alias dtype = DType.float32
    var ctx = DeviceContext()
    alias length: Int = 32
    alias blocks: Int = 8
    alias threads: Int = 4
    alias elements_in = blocks * threads

    # Step 1
    var in_buffer = ctx.enqueue_create_host_buffer[dtype](length)
    for idx in range(length):
        in_buffer[idx] = idx
    var out_buffer = ctx.enqueue_create_buffer[dtype](length)
    ctx.enqueue_copy(out_buffer, in_buffer)

    # Step 2
    alias layout = Layout.row_major(blocks, threads)
    var in_tensor = LayoutTensor[dtype, layout](in_buffer)
    alias InTensor = LayoutTensor[dtype, layout, MutableAnyOrigin]

    # Step 3
    var result_buffer = ctx.enqueue_create_buffer[dtype](8)
    _ = ctx.enqueue_memset(result_buffer, 0.0)
    alias result_layout = Layout.row_major(8)
    alias ResultTensor = LayoutTensor[dtype, result_layout, MutableAnyOrigin]

    var result_tensor = ResultTensor(result_buffer)

    # Debug:
    # with result_buffer.map_to_host() as host_buffer:
    #     print(host_buffer)
    # print(in_tensor)

    # Step 4
    # Kernel to perform sum-reduce

    fn warp_reduce_kernel(in_tensor: InTensor, out_tensor: ResultTensor):
        var value = in_tensor.load[1](block_idx.x, thread_idx.x)

        value = warp.sum(value)

        if block_idx.x == 0:
            print("thread: ", thread_idx.x, "value: ", value)

        if thread_idx.x == 0:
            out_tensor[block_idx.x] = value

    ctx.enqueue_function[warp_reduce_kernel](
        in_tensor, result_tensor, grid_dim=blocks, block_dim=threads
    )

    with result_buffer.map_to_host() as host_buffer:
        print("Input: ", in_buffer)
        print("Result: ", host_buffer)
