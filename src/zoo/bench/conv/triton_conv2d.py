import torch
import triton
import triton.language as tl

from ..registry import BENCH
from .act import get_act


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    H,
    W,
    C_in,
    C_out,
    K,
    H_out,
    W_out,
    x_stride_b,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    w_stride_co,
    w_stride_ci,
    w_stride_kh,
    w_stride_kw,
    y_stride_b,
    y_stride_co,
    y_stride_h,
    y_stride_w,
    padding,
    stride,
    BLOCK_HO: tl.constexpr,
    BLOCK_WO: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    batch = pid_bc // C_out
    cout = pid_bc % C_out

    offs_ho = tl.arange(0, BLOCK_HO)
    offs_wo = tl.arange(0, BLOCK_WO)

    ho = pid_h * BLOCK_HO + offs_ho[:, None]
    wo = pid_w * BLOCK_WO + offs_wo[None, :]

    ho_mask = ho < H_out
    wo_mask = wo < W_out
    out_mask = ho_mask & wo_mask

    acc = tl.zeros([BLOCK_HO, BLOCK_WO], dtype=tl.float32)

    for ci in range(C_in):
        for kh in range(K):
            for kw in range(K):
                hi = ho * stride + kh - padding
                wi = wo * stride + kw - padding

                hi_mask = (hi >= 0) & (hi < H)
                wi_mask = (wi >= 0) & (wi < W)
                in_mask = hi_mask & wi_mask & out_mask

                x_off = (
                    batch * x_stride_b
                    + ci * x_stride_c
                    + hi * x_stride_h
                    + wi * x_stride_w
                )
                x = tl.load(x_ptr + x_off, mask=in_mask, other=0.0)

                w_off = (
                    cout * w_stride_co
                    + ci * w_stride_ci
                    + kh * w_stride_kh
                    + kw * w_stride_kw
                )
                w = tl.load(w_ptr + w_off)

                acc += x.to(tl.float32) * w.to(tl.float32)

    if b_ptr is not None:
        bias = tl.load(b_ptr + cout).to(tl.float32)
        acc += bias

    y_off = batch * y_stride_b + cout * y_stride_co + ho * y_stride_h + wo * y_stride_w
    tl.store(y_ptr + y_off, acc, mask=out_mask)


def triton_conv2d(x, weight, bias=None, stride=1, padding=0):
    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_HO = 16
    BLOCK_WO = 16

    grid = (
        B * C_out,
        triton.cdiv(H_out, BLOCK_HO),
        triton.cdiv(W_out, BLOCK_WO),
    )

    conv2d_kernel[grid](
        x,
        weight,
        bias,
        y,
        H,
        W,
        C_in,
        C_out,
        K,
        H_out,
        W_out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        padding,
        stride,
        BLOCK_HO,
        BLOCK_WO,
    )

    return y


@triton.jit
def im2col_kernel(
    x_ptr,
    col_ptr,
    H,
    W,
    C,
    K,
    H_out,
    W_out,
    x_stride_b,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    col_stride_b,
    col_stride_ho,
    col_stride_wo,
    col_stride_ci,
    col_stride_kh,
    col_stride_kw,
    padding,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    total = C * K * K * H_out * W_out
    if pid >= total:
        return

    temp = pid
    kw = temp % K
    temp //= K
    kh = temp % K
    temp //= K
    wo_idx = temp % W_out
    temp //= W_out
    ho_idx = temp % H_out
    temp //= H_out
    ci = temp

    batch = 0

    ho = ho_idx
    wo = wo_idx

    hi = ho * stride + kh - padding
    wi = wo * stride + kw - padding

    if hi >= 0 and hi < H and wi >= 0 and wi < W:
        x_off = batch * x_stride_b + ci * x_stride_c + hi * x_stride_h + wi * x_stride_w
        val = tl.load(x_ptr + x_off)
    else:
        val = 0.0

    col_off = (
        batch * col_stride_b
        + ho_idx * col_stride_ho
        + wo_idx * col_stride_wo
        + ci * col_stride_ci
        + kh * col_stride_kh
        + kw * col_stride_kw
    )
    tl.store(col_ptr + col_off, val)


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = (offs_k + k) < K

        a = tl.load(a_ptrs + k * stride_ak, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs + k * stride_bk, mask=k_mask[:, None], other=0.0)

        acc += tl.dot(a, b)

    m_mask = offs_m < M
    n_mask = offs_n < N
    mask = m_mask[:, None] & n_mask[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask)


def triton_conv2d_im2col(x, weight, bias=None, stride=1, padding=0):
    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    col = torch.empty((B, H_out, W_out, C_in, K, K), device=x.device, dtype=x.dtype)
    col_2d = col.view(B * H_out * W_out, C_in * K * K)
    weight_2d = weight.view(C_out, C_in * K * K)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(C_in * K * K * H_out * W_out, BLOCK_SIZE),)

    im2col_kernel[grid](
        x,
        col,
        H,
        W,
        C_in,
        K,
        H_out,
        W_out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        col.stride(0),
        col.stride(1),
        col.stride(2),
        col.stride(3),
        col.stride(4),
        col.stride(5),
        padding,
        stride,
        BLOCK_SIZE,
    )

    y_2d = torch.empty((B * H_out * W_out, C_out), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    M = B * H_out * W_out
    N = C_out
    K_dim = C_in * K * K

    grid_mm = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid_mm](
        col_2d,
        weight_2d.T,
        y_2d,
        M,
        N,
        K_dim,
        col_2d.stride(0),
        col_2d.stride(1),
        weight_2d.T.stride(0),
        weight_2d.T.stride(1),
        y_2d.stride(0),
        y_2d.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    y = y_2d.view(B, H_out, W_out, C_out).permute(0, 3, 1, 2)

    if bias is not None:
        y = y + bias.view(1, -1, 1, 1)

    return y


class TritonConv2d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = torch.nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x):
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding)


@BENCH.register("triton_conv2d")
class BenchTritonConv2d(torch.nn.Module):
    def __init__(
        self,
        channels: int = 256,
        layers: int = 3,
        kernel_size: int = 3,
        act: str | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for _ in range(layers):
            self.weights.append(
                torch.nn.Parameter(
                    torch.randn(channels, channels, kernel_size, kernel_size)
                )
            )
            self.biases.append(torch.nn.Parameter(torch.randn(channels)))

        self.act = act
        if act == "relu":
            self.act_id = 1
        elif act == "gelu":
            self.act_id = 2
        elif act == "silu":
            self.act_id = 3
        else:
            self.act_id = 0

    def forward(self, x):
        for i in range(self.layers):
            x = triton_conv2d_im2col(
                x, self.weights[i], self.biases[i], 1, self.padding
            )
            if self.act is not None:
                act_fn = get_act(self.act)
                x = act_fn(x)
        return x
