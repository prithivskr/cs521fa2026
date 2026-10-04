import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler

# Create a log directory
logdir = "./jax_trace"


def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    """
    Reimplement the same function (im2col_manual) in myconv.py "for JAX".
    Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    """
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (P, P), (P, P)))

    patches = jnp.stack(
        [
            x_pad[:, :, kh : kh + out_h * S : S, kw : kw + out_w * S : S]
            for kh in range(KH)
            for kw in range(KW)
        ],
        axis=2,
    )
    return patches.reshape(N, C * KH * KW, out_h * out_w).transpose(0, 2, 1)


def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    """
    Reimplement the same function (conv2d_manual) in myconv.py "for JAX".
    Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    """
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    out_h = (H + 2 * padding - KH) // stride + 1
    out_w = (W + 2 * padding - KW) // stride + 1
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)
    weight_f = weight.reshape(C_out, -1)
    out = jnp.matmul(cols, weight_f.transpose(1, 0)) + bias
    return out.transpose(0, 2, 1).reshape(N, C_out, out_h, out_w)


if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 33, 33
    model = ConvModel(
        H, W, in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1
    )
    model.eval()

    # Example input
    x_torch = torch.randn(1, 3, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach()
        .cpu()
        .numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy(),  # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # enable JIT compilation
    conv2d_manual_jax_jit = jit(
        conv2d_manual_jax, static_argnames=("stride", "padding")
    )

    # call your JAX function
    out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)

    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print(
        "JAX --- correctness check:",
        np.allclose(
            np.asarray(out_jax), conv_ref.detach().numpy(), atol=1e-4, rtol=1e-5
        ),
    )
