from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from testing import assert_equal
from math import ceildiv

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
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    # Psuedocode convolution
    # for i in range(SIZE):
    #     for j in range(CONV):
    #         if i + j < SIZE:
    #             ret[i] += a_host[i + j] * b_host[j]
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    shared_a = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_b = tb[dtype]().row_major[TPB]().shared().alloc()

    if global_i < SIZE:
        shared_a[local_i] = a[global_i]

    if global_i < CONV:
        shared_b[local_i] = b[global_i]

    barrier()

    if global_i < SIZE:
        var local_sum: output.element_type = 0

        @parameter
        for j in range(CONV):
            if local_i + j < SIZE:
                local_sum += shared_a[local_i + j] * shared_b[j]
            barrier()

        output[global_i] = local_sum

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
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i  = thread_idx.x

    shared_a = tb[dtype]().row_major[TPB + CONV_2 - 1]().shared().alloc()  # input slice + halo
    shared_b = tb[dtype]().row_major[CONV_2]().shared().alloc()            # entire kernel

    if global_i < SIZE_2:                                # guard against out-of-range reads
        shared_a[local_i] = a[global_i]                  # coalesced load of main slice

    if local_i < CONV_2:                                 # only first CONV_2 threads participate
        shared_b[local_i] = b[local_i]                   # load kernel into shared memory

    if local_i < CONV_2 - 1:                             # threads responsible for halo load
        var next_idx = global_i + TPB                    # element that lives in next block
        shared_a[local_i + TPB] = a[next_idx] if next_idx < SIZE_2 else 0.0  # pad with zeros

    barrier()  # ensure shared memory is fully populated before computation

    if global_i < SIZE_2:                                # skip threads mapping past the end
        var local_sum: output.element_type = 0.0
        @parameter                                       # unroll small convolution
        for j in range(CONV_2):                          # dot product of window & kernel
            local_sum += shared_a[local_i + j] * shared_b[j]
        output[global_i] = local_sum                     # write result back to global memory

    # FILL ME IN (roughly 18 lines)


# ANCHOR_END: conv_1d_block_boundary


# ──────────────────────────────── constants
alias TPB_X       = 8                 # threads-per-block (x)
alias TPB_Y       = 8                 # threads-per-block (y)
alias WIDTH       = 16                # input image width
alias HEIGHT      = 12                # input image height
alias K           = 3                 # square kernel size
alias BLOCKS_PER_GRID_2D  = (ceildiv(WIDTH,  TPB_X), 
                              ceildiv(HEIGHT, TPB_Y))
alias THREADS_PER_BLOCK_2D = (TPB_X, TPB_Y)

alias in2d_layout    = Layout.row_major(HEIGHT, WIDTH)
alias out2d_layout   = Layout.row_major(HEIGHT, WIDTH)
alias kernel_layout  = Layout.row_major(K, K)
alias TILE_W = TPB_X + K - 1
alias TILE_H = TPB_Y + K - 1

# ──────────────────────────────── kernel
fn conv_2d_halo[
    in_layout : Layout, out_layout : Layout,
    k_layout  : Layout, dtype : DType
](
    output  : LayoutTensor[mut = False, dtype, out_layout],
    inp     : LayoutTensor[mut = False, dtype, in_layout],
    kernel  : LayoutTensor[mut = False, dtype, k_layout],
):
    var gx  = block_idx.x * block_dim.x + thread_idx.x      # global X/Y
    var gy  = block_idx.y * block_dim.y + thread_idx.y
    var lx  = thread_idx.x                                   # local  X/Y
    var ly  = thread_idx.y

    # tile dim  = block + halo

    # shared image tile + shared kernel
    shared_img = tb[dtype]().row_major[TILE_H, TILE_W]().shared().alloc()
    shared_k   = tb[dtype]().row_major[K, K]().shared().alloc()

    # ───── phase-1 : main tile load (perfectly coalesced)
    if gx < WIDTH and gy < HEIGHT:
        shared_img[ly, lx] = inp[gy, gx]
    else:
        shared_img[ly, lx] = 0.0     # off-chip → pad zero

    # ───── phase-2 : halo load
    # Every thread that owns a halo slot loads it; stride == blockDim.* so no bank fights
    var halo_y = ly
    while halo_y < TILE_H:
        var halo_x = lx
        var gyy = block_idx.y * block_dim.y + halo_y
        while halo_x < TILE_W:
            var gxx = block_idx.x * block_dim.x + halo_x
            shared_img[halo_y, halo_x] = (
                inp[gyy, gxx] if (gxx < WIDTH and gyy < HEIGHT) else 0.0
            )
            halo_x += TPB_X
        halo_y += TPB_Y

    # ───── phase-3 : cache the K×K kernel – first K² threads only
    # let exactly the K×K threads (ly in [0,K), lx in [0,K)) load each weight
    if ly < K and lx < K:
        shared_k[ly, lx] = kernel[ly, lx]
    # if ly * TPB_X + lx < K * K:
    #     shared_k[ly, lx] = kernel[ly, lx]
    barrier()                                   # make everything visible

    # ───── compute
    if gx < WIDTH and gy < HEIGHT:
        var sum: output.element_type = 0.0
        @parameter 
        for ky in range(K):
            @parameter 
            for kx in range(K):
                sum += shared_img[ly + ky, lx + kx] * shared_k[ky, kx]
        output[gy, gx] = sum


def main():
    with DeviceContext() as ctx:
        size = SIZE_2 if argv()[1] == "--block-boundary" else SIZE
        conv = CONV_2 if argv()[1] == "--block-boundary" else CONV
        a = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](conv).enqueue_fill(0)
        img = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH).enqueue_fill(0)
        ker = ctx.enqueue_create_buffer[dtype](K * K).enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        with b.map_to_host() as b_host:
            for i in range(conv):
                b_host[i] = i
        with img.map_to_host() as img_host:
            for i in range(HEIGHT * WIDTH):
                img_host[i] = 1.0

        with ker.map_to_host() as ker_host:
            for i in range(K * K):
                ker_host[i] = 1.0

        if argv()[1] == "--simple":
            out = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)
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
            out = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)
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
        elif argv()[1] == "--conv2d":
            out = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH).enqueue_fill(0)
            var out_t = LayoutTensor[mut=False, dtype, out2d_layout](out.unsafe_ptr())
            var in_t  = LayoutTensor[mut=False, dtype, in2d_layout](img.unsafe_ptr())
            var k_t   = LayoutTensor[mut=False, dtype, kernel_layout](ker.unsafe_ptr())

            ctx.enqueue_function[
                conv_2d_halo[in2d_layout, out2d_layout, kernel_layout, dtype]
            ](
                out_t, in_t, k_t,
                grid_dim  = BLOCKS_PER_GRID_2D,
                block_dim = THREADS_PER_BLOCK_2D,
            )

        else:
            raise Error("Invalid argument")

        ctx.synchronize()

        if argv()[1] == "--conv2d":
            expected_size = HEIGHT * WIDTH
            expected = ctx.enqueue_create_host_buffer[dtype](expected_size).enqueue_fill(0)
            with img.map_to_host() as h_img, ker.map_to_host() as h_ker:
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        var acc: Float32  = Float32(0.0)
                        for ky in range(K):
                            for kx in range(K):
                                var iy = y + ky
                                var ix = x + kx
                                if iy < HEIGHT and ix < WIDTH:
                                    acc += h_img[iy * WIDTH + ix] * h_ker[ky * K + kx]
                        expected[y * WIDTH + x] = acc
        else:
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