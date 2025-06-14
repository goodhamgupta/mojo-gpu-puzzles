from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite


alias SIZE = 128
alias TPB = 128
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias layout = Layout.row_major(SIZE)
alias dtype = DType.float32


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
):
    global_id = block_dim.x * block_idx.x + thread_idx.x
    local_id = thread_idx.x
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()

    # Step 1: Find max
    # var thread_max: Scalar[dtype] = min_finite[dtype]()
    if global_id < input_size:
        # thread_max = rebind[Scalar[dtype]](input[global_id])

        # Note: I'm a bit confused about this approach.
        # By default, it looks like we are just copying over the
        # value in the input to the shared memory.
        # For the current thread, it's own value is considered as "max" for now
        # Instead of assigning it to another variable and doing the
        # "rebind" shenanigans, we could just assign it directly to the shared memory?
        # Turns out, this also works fine. I see minor changes in running time(from ~0.306s to ~0.313s),
        # but the overall implementation is still correct. 💪
        # I'm not sure why. There is no redundant memory access pattern here from what I can tell.
        shared_max[local_id] = input[global_id]

    # shared_max[local_id] = thread_max
    barrier()

    stride = TPB // 2

    # Parallel reduction to find max
    while stride > 0:
        if local_id < input_size:
            shared_max[local_id] = max(
                shared_max[local_id], shared_max[local_id + stride]
            )
        barrier()
        stride //= 2

    block_max = shared_max[0]

    # Step 2: Compute exponential values using the numerically stable formula
    var exp_val: out.element_type = 0.0

    if global_id < input_size:
        exp_val = rebind[Scalar[dtype]](exp(input[global_id] - block_max))
        out[global_id] = exp_val
        # NOTE: Here, directly assigning the value to the output DOESN'T WORK
        # It gives only nan/inf values
        # out[global_id] = exp(input[global_id] - block_max)

    shared_sum[local_id] = exp_val
    barrier()

    stride = TPB // 2

    # Parallel reduction to find sum
    while stride > 0:
        if local_id < input_size:
            shared_sum[local_id] += shared_sum[local_id + stride]
        barrier()
        stride //= 2

    block_sum = shared_sum[0]

    if global_id < input_size:
        out[global_id] = out[global_id] / block_sum

    # FILL IN (roughly 31 lines)
    ...


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    # Step 1: Find maximum element
    var max_val: out.element_type = 0.0
    var running_sum: out.element_type = 0.0

    # NOTE: Couldn't see any noticeable improvement in the runnning time after adding the `parameter` decorator.
    # I should measure performance more accurately.
    # Also, there's a chance that at the current matrix size, there's not much perf improvement with loop unrolling.
    @parameter
    for idx in range(input_size):
        if input[idx] > max_val:
            max_val = input[idx]

    # Step 2: Find softmax
    @parameter
    for idx in range(input_size):
        out[idx] = exp(input[idx] - max_val)
        running_sum += out[idx]

    # Step 3: Compute final values
    @parameter
    for idx in range(input_size):
        out[idx] = out[idx] / running_sum

    # FILL IN (roughly 10 lines)
    ...


# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](input.to_layout_tensor())
        alias layout = input_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            gpu_ctx.enqueue_function[
                softmax_gpu_kernel[layout, input_size, dtype]
            ](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=(TPB, 1),
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
