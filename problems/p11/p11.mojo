from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from testing import assert_equal

# ANCHOR: conv_1d_simple
alias TPB = 8
alias SIZE = 6
alias CONV = 3
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias in_layout = Layout.row_major(SIZE)
alias out_layout = Layout.row_major(SIZE)
alias conv_layout = Layout.row_major(CONV)


fn conv_1d_simple[
    in_layout: Layout, out_layout: Layout, conv_layout: Layout
](
    out: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    # Psuedocode convolution
    # for i in range(SIZE):
    #   for j in range(CONV):
    #     if i + j < SIZE:
    #         ret[i] += a_host[i + j] * b_host[j]
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    a_size = a.shape[0]()
    b_size = b.shape[0]()
    shared_size = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_conv = tb[dtype]().row_major[TPB]().shared().alloc()
    # Optimisation: Change the memory of shared_conv to be of size CONV
    # and then assign using global_i
    # shared_conv = tb[dtype]().row_major[CONV]().shared().alloc()
    # LayoutTensor will ensure that the program doesn't crash if global_i > CONV

    if global_i < SIZE:
        shared_size[local_i] = a[global_i]

        # Optimisation:
        # Assign the shared_conv here, and let layouttensor take care of ignoring
        # invalid indices.
        shared_conv[local_i] = b[global_i]
    # if global_i < CONV:
    #     shared_conv[local_i] = b[global_i]

    barrier()

    if global_i < SIZE:
        var local_sum: out.element_type = 0

        # This unrolls the loop
        # https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-for-statement
        @parameter
        for j in range(CONV):
            if local_i + j < SIZE:
                local_sum += shared_size[local_i + j] * shared_conv[j]

        out[global_i] = local_sum
    # FILL ME IN (roughly 14 lines)


# ANCHOR_END: conv_1d_simple

# ANCHOR: conv_1d_block_boundary
alias SIZE_2 = 15
alias CONV_2 = 4
alias BLOCKS_PER_GRID_2 = (2, 1)
alias THREADS_PER_BLOCK_2 = (TPB, 1)
alias in_2_layout = Layout.row_major(SIZE_2)
alias out_2_layout = Layout.row_major(SIZE_2)
alias conv_2_layout = Layout.row_major(CONV_2)


fn conv_1d_block_boundary[
    in_layout: Layout, out_layout: Layout, conv_layout: Layout, dtype: DType
](
    out: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # step 1: account for padding
    shared_a = tb[dtype]().row_major[TPB + CONV_2 - 1]().shared().alloc()
    shared_b = tb[dtype]().row_major[CONV_2]().shared().alloc()

    if global_i < SIZE_2:
        shared_a[local_i] = a[global_i]

    # step 2: load elements needed for convolution at block boundary
    # At the block boundary, in the worst case, we only need 3 more elements
    # from the next block(because the size of the conv filter is 4)
    # Eg:
    # global_i = 2, local_i = 2
    # next_idx = 2 + 8 = 10
    # if 10 < 15, TRUE
    # TRUE: shared_a[8 + 2] = a[10]
    # Q: Do we need this? Can I not just do
    # if global_i < SIZE_2: shared_a[local_i] = a[global_i]
    # and let LayoutTensor handle the OOB case?
    if local_i < CONV_2 - 1:
        next_idx = global_i + TPB
        if next_idx < SIZE_2:
            shared_a[TPB + local_i] = a[next_idx]
        else:
            # init OOB elemnts with 0
            shared_a[TPB + local_i] = 0

    if local_i < CONV_2:
        shared_b[local_i] = b[local_i]

    barrier()

    if global_i < SIZE_2:
        var local_sum: out.element_type = 0

        @parameter
        for j in range(CONV_2):
            if local_i + j < TPB + CONV - 1:
                local_sum += shared_a[local_i + j] * shared_b[j]
        
        out[global_i] = local_sum

    # FILL ME IN (roughly 18 lines)


# ANCHOR_END: conv_1d_block_boundary


def main():
    with DeviceContext() as ctx:
        size = SIZE_2 if argv()[1] == "--block-boundary" else SIZE
        conv = CONV_2 if argv()[1] == "--block-boundary" else CONV
        out = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](conv).enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        with b.map_to_host() as b_host:
            for i in range(conv):
                b_host[i] = i

        if argv()[1] == "--simple":
            var out_tensor = LayoutTensor[mut=False, dtype, out_layout](
                out.unsafe_ptr()
            )
            var a_tensor = LayoutTensor[mut=False, dtype, in_layout](
                a.unsafe_ptr()
            )
            var b_tensor = LayoutTensor[mut=False, dtype, conv_layout](
                b.unsafe_ptr()
            )
            ctx.enqueue_function[
                conv_1d_simple[in_layout, out_layout, conv_layout]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--block-boundary":
            var out_tensor = LayoutTensor[mut=False, dtype, out_2_layout](
                out.unsafe_ptr()
            )
            var a_tensor = LayoutTensor[mut=False, dtype, in_2_layout](
                a.unsafe_ptr()
            )
            var b_tensor = LayoutTensor[mut=False, dtype, conv_2_layout](
                b.unsafe_ptr()
            )
            ctx.enqueue_function[
                conv_1d_block_boundary[
                    in_2_layout, out_2_layout, conv_2_layout, dtype
                ]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
        else:
            raise Error("Invalid argument")

        ctx.synchronize()
        expected = ctx.enqueue_create_host_buffer[dtype](size).enqueue_fill(0)

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(size):
                for j in range(conv):
                    if i + j < size:
                        expected[i] += a_host[i + j] * b_host[j]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(size):
                for j in range(conv):
                    if i + j < size:
                        assert_equal(out_host[i], expected[i])
