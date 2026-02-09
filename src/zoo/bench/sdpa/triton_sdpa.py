#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import triton
import triton.language as tl

from ..registry import BENCH


@triton.jit
def sdpa(
    q_ptr,  # BxHxL_qxD_qk
    k_ptr,  # BxHxL_kvxD_qk
    v_ptr,  # BxHxL_kvxD_v
    a_ptr,  # BxHxL_qxD_v
    H,
    L_q,
    L_kv,
    D_qk,
    D_v,
    q_stride_h,
    q_stride_l,
    q_stride_d,
    k_stride_h,
    k_stride_l,
    k_stride_d,
    v_stride_h,
    v_stride_l,
    v_stride_d,
    a_stride_h,
    a_stride_l,
    a_stride_d,
    scale: float,
    BLOCK_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # most outter loop
    pid_q = tl.program_id(1)
    # Q[..., qi:qi+BQ, :]
    q_beg = pid_bh * q_stride_h + pid_q * BLOCK_Q * q_stride_l
    q_beg += q_stride_l * tl.arange(0, BLOCK_Q)[None, :, None]
    max_qk = tl.zeros([1, BLOCK_Q, 1], dtype=tl.float32) - float("inf")
    sum_exp = tl.zeros([1, BLOCK_Q, 1], dtype=tl.float32)
    for i_kv in range(0, tl.cdiv(L_kv, BLOCK_KV)):
        # K[..., ki:ki+BK, :]
        k_beg = pid_bh * k_stride_h + i_kv * BLOCK_KV * k_stride_l
        k_beg += k_stride_l * tl.arange(0, BLOCK_KV)[None, :, None]
        # S = QK
        s_acc = tl.zeros([1, BLOCK_Q, BLOCK_KV], dtype=tl.float32)
        for d in range(tl.cdiv(D_qk, BLOCK_D)):
            q_off = q_beg + (d + tl.arange(0, BLOCK_D)[None, None, :]) * q_stride_d
            q = tl.load(q_ptr + q_off)
            k_off = k_beg + (d + tl.arange(0, BLOCK_D)[None, None, :]) * k_stride_d
            k = tl.load(k_ptr + k_off)
            s_acc = tl.dot(q, tl.permute(k, [0, 2, 1]), s_acc)
        s_acc *= scale
        s_rowmax = tl.max(s_acc, axis=-1, keep_dims=True)  # [1, Q, 1]
        s_acc = tl.exp(s_acc - s_rowmax)
        s_sum = tl.sum(s_acc, axis=-1, keep_dims=True)  # [1, Q, 1]
        max_qk_new = tl.maximum(s_rowmax, max_qk)
        sum_exp_new = sum_exp * tl.exp(max_qk - max_qk_new)
        sum_exp_new += s_sum * tl.exp(s_rowmax - max_qk_new)
        # A = SV
        v_beg = pid_bh * v_stride_h + i_kv * BLOCK_KV * v_stride_l
        v_beg += v_stride_l * tl.arange(0, BLOCK_KV)[None, :, None]
        a_beg = pid_bh * a_stride_h + pid_q * BLOCK_Q * a_stride_l
        a_beg += a_stride_l * tl.arange(0, BLOCK_Q)[None, :, None]
        for d in range(tl.cdiv(D_v, BLOCK_D)):
            v_off = v_beg + (d + tl.arange(0, BLOCK_D)[None, None, :]) * v_stride_d
            v = tl.load(v_ptr + v_off)
            a_off = a_beg + (d + tl.arange(0, BLOCK_D)[None, None, :]) * a_stride_d
            attn = tl.load(a_ptr + a_off).to(tl.float32)
            attn *= tl.exp(max_qk - max_qk_new) * sum_exp / sum_exp_new
            attn = tl.dot(s_acc.to(tl.float16), v, attn)
            attn *= tl.exp(s_rowmax - max_qk_new) / sum_exp_new
            tl.store(a_ptr + a_off, attn.to(tl.float16))
        sum_exp = sum_exp_new
        max_qk = max_qk_new


@BENCH.register("triton_sdpa")
class BenchTritonSDPA(torch.nn.Module):
    def __init__(self, block_q: int = 16, block_kv: int = 16, block_d: int = 16):
        super().__init__()
        self.block_q = block_q
        self.block_kv = block_kv
        self.block_d = block_d
        self.a = None

    def forward(self, hidden_states):
        *_, H, L, D = hidden_states.shape
        hidden_states = hidden_states.view(-1, L, D)
        H, L, D = hidden_states.shape
        if self.a is None:
            self.a = torch.empty_like(hidden_states)

        def grid(META):
            return (H, triton.cdiv(L, META["BLOCK_Q"]))

        sdpa[grid](
            hidden_states,
            hidden_states,
            hidden_states,
            self.a,
            H,
            L,
            L,
            D,
            D,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            self.a.stride(0),
            self.a.stride(1),
            self.a.stride(2),
            scale=1,
            BLOCK_Q=self.block_q,
            BLOCK_KV=self.block_kv,
            BLOCK_D=self.block_d,
        )
        return self.a
