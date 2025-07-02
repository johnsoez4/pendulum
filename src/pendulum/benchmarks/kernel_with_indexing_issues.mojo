from gpu import thread_idx, block_dim, block_idx
from layout import Layout, LayoutTensor

alias dtype = DType.float32
# alias dtype = Float32


# GPU kernel with tensor indexing issues (for demonstration)
fn gpu_neural_network_kernel_with_indexing_issues(
    output_buffer: LayoutTensor[mut=True, dtype, Layout.row_major(1, 3)],
    input_buffer: LayoutTensor[mut=True, dtype, Layout.row_major(1, 4)],
):
    """GPU neural network kernel that demonstrates tensor indexing type conversion issues.
    """
    idx = thread_idx.x + block_idx.x * block_dim.x

    if idx < 3:  # 3 outputs
        # Compute neural network output exactly like CPU version
        # Use identical weight formula: (i + j + 1) * 0.1
        var sum: Float32 = 0.0

        # This will cause tensor indexing type conversion issues
        @parameter
        for j in range(4):  # 4 inputs
            weight = Float32(idx + j + 1) * 0.1
            sum = sum + input_buffer[0, j][0] * weight

        # Apply real tanh activation
        var tanh_result: Float32
        if sum > 5.0:
            tanh_result = 1.0
        elif sum < -5.0:
            tanh_result = -1.0
        else:
            # High-quality tanh approximation
            abs_sum = sum if sum >= 0.0 else -sum
            sum_squared = sum * sum
            denominator = 1.0 + abs_sum / 3.0 + sum_squared / 15.0
            tanh_result = sum / denominator

        # Store result - this may also have indexing issues
        output_buffer[0, idx] = tanh_result
