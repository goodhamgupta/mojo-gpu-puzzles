from sys import sizeof, argv
from testing import assert_equal
from gpu.host import DeviceContext
from gpu.memory import async_copy_wait_all

# ANCHOR: naive_matmul
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from layout.tensor_builder import LayoutTensorBuild as tb


alias TPB = 3
alias SIZE = 2
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, TPB)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)


fn naive_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x

    if row < SIZE and col < SIZE:
        var running_sum: out.element_type = 0

        @parameter
        for k in range(SIZE):
            # NOTE: Looks like a race condition occurs here.
            # Without the print statement, the last element doesn't match the expected value.
            # HOWEVER, with it, the values match!
            # I tried creating a varible to track the "running_sum", and then assign it, but
            # I ran into more type errors.
            # FIX: Use out.element_type instead of dtype for the variable
            running_sum += a[row, k] * b[k, col]
        out[row, col] = running_sum


# ANCHOR_END: naive_matmul


# ANCHOR: single_block_matmul
fn single_block_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    local_row = thread_idx.y
    local_col = thread_idx.x
    shared_a = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    shared_b = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    if row < size and col < size and local_row < size and local_col < size:
        shared_a[local_row, local_col] = a[row, col]
        shared_b[local_row, local_col] = b[row, col]

    barrier()
    if row < size and col < size and local_row < size and local_col < size:
        var running_sum: out.element_type = 0.0

        @parameter
        for k in range(size):
            running_sum += a[local_row, k] * b[k, local_col]

        out[row, col] = running_sum
    # FILL ME IN (roughly 12 lines)


# ANCHOR_END: single_block_matmul

# ANCHOR: matmul_tiled
alias SIZE_TILED = 8
alias BLOCKS_PER_GRID_TILED = (3, 3)  # each block convers 3x3 elements
alias THREADS_PER_BLOCK_TILED = (TPB, TPB)
alias layout_tiled = Layout.row_major(SIZE_TILED, SIZE_TILED)


fn matmul_tiled[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    # LayoutTensor APIs
    out_tile = out.tile[TPB, TPB](block_idx.y, block_idx.x)
    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    local_row = thread_idx.y
    local_col = thread_idx.x

    var acc: out.element_type = 0.0

    alias load_a_layout = Layout.row_major(1, TPB)
    alias load_b_layout = Layout.row_major(TPB, 1)
    for idx in range((size + TPB - 1) // TPB):
        a_tile = a.tile[TPB, TPB](block_idx.y, idx)
        b_tile = b.tile[TPB, TPB](idx, block_idx.x)

        copy_dram_to_sram_async[thread_layout=load_a_layout](a_shared, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_shared, b_tile)

        async_copy_wait_all()

        barrier()

        @parameter
        for k in range(TPB):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

        if (
            block_idx.y * TPB + local_row < size
            and block_idx.x * TPB + local_col < size
        ):
            out_tile[local_row, local_col] = acc

    # Manual memory management
    # local_row = thread_idx.y
    # local_col = thread_idx.x
    # tiled_row = block_idx.y * TPB + local_row
    # tiled_col = block_idx.x * TPB + local_col

    # # Shared memory
    # shared_a = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    # shared_b = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    # var acc: out.element_type = 0.0

    # @parameter
    # for tile in range((SIZE_TILED + TPB - 1) // TPB):
    #     if local_row < TPB and local_col < TPB:
    #         shared_a[local_row, local_col] = 0
    #         shared_b[local_row, local_col] = 0
    #     barrier()

    #     # load tile A
    #     if tiled_row < SIZE_TILED and (tile * TPB + local_col) < SIZE_TILED:
    #         shared_a[local_row, local_col] = a[
    #             tiled_row, tile * TPB + local_col
    #         ]

    #     if (tile * TPB + local_row) < SIZE_TILED and tiled_col < SIZE_TILED:
    #         shared_b[local_row, local_col] = b[
    #             tile * TPB + local_row, tiled_col
    #         ]
    #     barrier()

    #     if tiled_row < SIZE_TILED and tiled_col < SIZE_TILED:

    #         @parameter
    #         for k in range(min(TPB, SIZE_TILED - tile * TPB)):
    #             acc += shared_a[local_row, k] * shared_b[k, local_col]
    #     barrier()

    # if tiled_row < SIZE_TILED and tiled_col < SIZE_TILED:
    #     out[tiled_row, tiled_col] = acc

    # FILL ME IN (roughly 20 lines)


# ANCHOR_END: matmul_tiled


def main():
    with DeviceContext() as ctx:
        size = SIZE_TILED if argv()[1] == "--tiled" else SIZE
        out = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp1 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp2 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](
            size * size
        ).enqueue_fill(0)
        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(size):
                for col in range(size):
                    val = row * size + col
                    # row major: placing elements row by row
                    inp1_host[row * size + col] = val
                    inp2_host[row * size + col] = Float32(2.0) * val

            # inp1 @ inp2.T
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        expected[i * size + j] += (
                            inp1_host[i * size + k] * inp2_host[k * size + j]
                        )

        out_tensor = LayoutTensor[mut=False, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](inp1.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](inp2.unsafe_ptr())

        if argv()[1] == "--naive":
            ctx.enqueue_function[naive_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--single-block":
            ctx.enqueue_function[single_block_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--tiled":
            # Need to update the layout of the tensors to the tiled layout
            out_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                out.unsafe_ptr()
            )
            a_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp1.unsafe_ptr()
            )
            b_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp2.unsafe_ptr()
            )

            ctx.enqueue_function[matmul_tiled[layout_tiled, SIZE_TILED]](
                out_tensor_tiled,
                a_tensor_tiled,
                b_tensor_tiled,
                grid_dim=BLOCKS_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )
        else:
            raise Error("Invalid argument")

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for col in range(size):
                for row in range(size):
                    assert_equal(
                        out_host[col * size + row], expected[col * size + row]
                    )
