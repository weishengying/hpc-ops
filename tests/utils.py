import torch


def calculate_errors(ref_tensor, real_tensor, eps=1e-6, top_k=10):
    """
    Calculate various error metrics between reference and real tensors

    Args:
        ref_tensor: Reference tensor (PyTorch tensor)
        real_tensor: Real tensor (PyTorch tensor with the same shape as ref_tensor)
        eps: Small value to prevent division by zero, default 1e-6
        top_k: Number of top largest errors to return, default 10

    Returns:
        dict: Dictionary containing the following metrics:
            - mean_abs_error: Mean Absolute Error
            - max_abs_error: Maximum Absolute Error
            - max_abs_error_ref: Reference value at the position of maximum absolute error
            - max_abs_error_real: Real value at the position of maximum absolute error
            - max_abs_error_pos: Position coordinates of maximum absolute error (as tuple)
            - mean_rel_error: Mean Relative Error
            - max_rel_error: Maximum Relative Error
            - max_rel_error_ref: Reference value at the position of maximum relative error
            - max_rel_error_real: Real value at the position of maximum relative error
            - max_rel_error_pos: Position coordinates of maximum relative error (as tuple)
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(ref_tensor, torch.Tensor) or not isinstance(real_tensor, torch.Tensor):
        raise TypeError("Inputs must be PyTorch tensors")

    # Check if tensor shapes match
    if ref_tensor.shape != real_tensor.shape:
        raise ValueError("Reference and real tensors must have the same shape")

    # Calculate absolute errors
    abs_error = torch.abs(ref_tensor - real_tensor)

    # Mean Absolute Error
    mae = torch.mean(abs_error).item()

    # Get top K absolute errors and their positions
    num_elements = abs_error.numel()
    k = min(top_k, num_elements)

    # Flatten the error tensor and obtain the indices of the top k largest values
    abs_error_flat = abs_error.flatten()
    top_abs_values, top_abs_flat_indices = torch.topk(abs_error_flat, k, largest=True)

    # Convert to multidimensional coordinates and collect corresponding values
    top_abs_errors = []
    for val, idx in zip(top_abs_values, top_abs_flat_indices):
        pos = tuple(torch.unravel_index(idx, abs_error.shape))
        top_abs_errors.append(
            {
                "error_value": val.item(),
                "ref_value": ref_tensor[pos].item(),
                "real_value": real_tensor[pos].item(),
                "position": pos,
            }
        )

    # Calculate relative errors (with protection against division by zero)
    rel_error = abs_error / (torch.max(torch.abs(ref_tensor), torch.abs(real_tensor)) + eps)

    # Mean Relative Error
    mre = torch.mean(rel_error).item()

    # Get top K relative errors and their positions
    rel_error_flat = rel_error.flatten()
    top_rel_values, top_rel_flat_indices = torch.topk(rel_error_flat, k, largest=True)

    # Convert to multidimensional coordinates and collect corresponding values
    top_rel_errors = []
    for val, idx in zip(top_rel_values, top_rel_flat_indices):
        pos = tuple(torch.unravel_index(idx, rel_error.shape))
        top_rel_errors.append(
            {
                "error_value": val.item(),
                "ref_value": ref_tensor[pos].item(),
                "real_value": real_tensor[pos].item(),
                "position": pos,
            }
        )

    return {
        "mean_abs_error": mae,
        "top_abs_errors": top_abs_errors,
        "mean_rel_error": mre,
        "top_rel_errors": top_rel_errors,
    }


def errors_to_string(error_results, precision=6):
    """
    Convert error calculation results to a human-readable string

    Args:
        error_results: Dictionary returned by calculate_errors function
        precision: Number of decimal places to display, default 6

    Returns:
        str: Formatted string with error information
    """
    # Create the header section
    lines = [""]
    lines.append("=" * 80)
    lines.append("Error Analysis Results".center(80))
    lines.append("=" * 80)
    lines.append("")

    # Add mean error metrics
    lines.append("Mean Error Metrics:")
    lines.append("-" * 40)
    lines.append(f"Mean Absolute Error: {error_results['mean_abs_error']:.{precision}f}")
    lines.append(f"Mean Relative Error: {error_results['mean_rel_error']:.{precision}f}")
    lines.append("")

    # Add top absolute errors section
    lines.append(f"Top {len(error_results['top_abs_errors'])} Absolute Errors:")
    lines.append("-" * 80)
    # Header for the table
    abs_header = (
        f"Rank".ljust(6)
        + f"Error Value".ljust(16)
        + f"Ref Value".ljust(16)
        + f"Real Value".ljust(16)
        + f"Position"
    )
    lines.append(abs_header)
    lines.append("-" * 80)

    # Add each top absolute error
    for i, err in enumerate(error_results["top_abs_errors"], 1):
        line = (
            f"{i:^6}"
            + f"{err['error_value']:.{precision}f}".ljust(16)
            + f"{err['ref_value']:.{precision}f}".ljust(16)
            + f"{err['real_value']:.{precision}f}".ljust(16)
            + f"{err['position']}"
        )
        lines.append(line)
    lines.append("")

    # Add top relative errors section
    lines.append(f"Top {len(error_results['top_rel_errors'])} Relative Errors:")
    lines.append("-" * 80)
    # Header for the table
    rel_header = (
        f"Rank".ljust(6)
        + f"Error Value".ljust(16)
        + f"Ref Value".ljust(16)
        + f"Real Value".ljust(16)
        + f"Position"
    )
    lines.append(rel_header)
    lines.append("-" * 80)

    # Add each top relative error
    for i, err in enumerate(error_results["top_rel_errors"], 1):
        line = (
            f"{i:^6}"
            + f"{err['error_value']:.{precision}f}".ljust(16)
            + f"{err['ref_value']:.{precision}f}".ljust(16)
            + f"{err['real_value']:.{precision}f}".ljust(16)
            + f"{err['position']}"
        )
        lines.append(line)
    lines.append("")

    lines.append("=" * 80)

    # Join all lines into a single string
    return "\n".join(lines)


def allclose(ref_tensor, real_tensor, atol=1e-8, rtol=1e-5):
    assert ref_tensor.dtype == real_tensor.dtype
    assert ref_tensor.device == real_tensor.device
    assert ref_tensor.shape == real_tensor.shape
    is_true = torch.allclose(
        ref_tensor.to(torch.float32), real_tensor.to(torch.float32), atol=atol, rtol=rtol
    )
    if not is_true:
        print(
            errors_to_string(
                calculate_errors(ref_tensor.to(torch.float32), real_tensor.to(torch.float32))
            )
        )
    return is_true


FP8_MAX = 448
FP4_MAX = 6.0
FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _ue8m0_scale(amax, value_max):
    """Per-block UE8M0 (power-of-two) dequant scale, stored as float8_e8m0fnu."""
    scale = (amax / value_max).clamp_min(1e-30).log2().ceil().exp2()
    return scale.to(torch.float8_e8m0fnu)


def quantize_mxfp8(w, block=32):
    """MXFP8 quantization along the last dim: fp8-e4m3 elements + UE8M0 scale."""
    shape = w.shape
    blocks = w.reshape(*shape[:-1], shape[-1] // block, block)
    scale = _ue8m0_scale(blocks.abs().amax(dim=-1, keepdim=True), FP8_MAX)
    w_fp8 = (blocks / scale.to(torch.float32)).reshape(shape).to(torch.float8_e4m3fn)
    return w_fp8, scale.squeeze(-1)


def dequantize_mxfp8(w_fp8, scale, block=32):
    """Inverse of quantize_mxfp8 -> fp32 tensor of w_fp8's shape."""
    shape = w_fp8.shape
    blocks = w_fp8.to(torch.float32).reshape(*shape[:-1], shape[-1] // block, block)
    return (blocks * scale.to(torch.float32).unsqueeze(-1)).reshape(shape)


def _fp4_encode(x):
    levels = FP4_LEVELS.to(x.device)
    bounds = (levels[1:] + levels[:-1]) / 2
    mag_idx = torch.bucketize(x.abs(), bounds).to(torch.uint8)
    sign_bit = (x < 0).to(torch.uint8)
    return (sign_bit << 3) | mag_idx


def _fp4_decode(code):
    levels = FP4_LEVELS.to(code.device)
    mag = levels[(code & 0x7).long()]
    sign = 1.0 - 2.0 * ((code >> 3) & 0x1).to(torch.float32)
    return sign * mag


def quantize_mxfp4(w, block=32):
    """MXFP4 quantization along the last dim, packed two values per uint8."""
    shape = w.shape
    blocks = w.reshape(*shape[:-1], shape[-1] // block, block)
    scale = _ue8m0_scale(blocks.abs().amax(dim=-1, keepdim=True), FP4_MAX)
    codes = _fp4_encode((blocks / scale.to(torch.float32)).reshape(shape))
    lo, hi = codes[..., 0::2], codes[..., 1::2]
    packed = (hi << 4) | lo
    return packed, scale.squeeze(-1)


def dequantize_mxfp4(packed, scale, block=32):
    """Inverse of quantize_mxfp4 -> fp32 tensor with last dim K = 2 * packed_K."""
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
    values = _fp4_decode(codes)
    shape = values.shape
    blocks = values.reshape(*shape[:-1], shape[-1] // block, block)
    return (blocks * scale.to(torch.float32).unsqueeze(-1)).reshape(shape)
