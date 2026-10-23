"""Memory-efficient cost-volume kernel (sampling + correlation) in Triton.

The fused TRT segment materializes the warped tensor (~8 GiB fp32) and needs
~2.4 GiB of tactic scratch. This kernel computes each output row (one
program per (view, depth, y)) with an online reduction over source views
and channels, so no warped tensor ever materializes. Row-vector loads keep
the gather coalesced along the contiguous W axis (~0.15 s at the serving
geometry vs ~0.85 s for the scalar variant). ``grid`` is the normalized
sampling coordinate map produced by ``warp_with_pose_depth_candidates``
(unchanged math).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def cost_volume_kernel(
    ref_ptr,
    tgt_ptr,
    grid_ptr,
    out_ptr,
    K: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    bv = tl.program_id(0)
    d = tl.program_id(1)
    h = tl.program_id(2)
    ws = tl.arange(0, BLOCK_W)
    wmask = ws < W

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for k in range(K):
        gbase = ((bv * K + k) * D * H * W + d * H * W + h * W) * 2
        gx = tl.load(grid_ptr + gbase + ws * 2, mask=wmask, other=0.0)
        gy = tl.load(grid_ptr + gbase + ws * 2 + 1, mask=wmask, other=0.0)
        # saturate fp16-Inf coords (behind-camera rays overflow the fp16
        # grid chain): Inf-Inf bilinear weights would poison the acc; any
        # finite far-out coordinate masks to zero via corner bounds
        gx = tl.minimum(tl.maximum(gx, -65504.0), 65504.0)
        gy = tl.minimum(tl.maximum(gy, -65504.0), 65504.0)
        x = (gx + 1) * (W - 1) / 2
        y = (gy + 1) * (H - 1) / 2
        x0f, y0f = tl.floor(x), tl.floor(y)
        x1f, y1f = x0f + 1, y0f + 1
        wx0, wx1 = x1f - x, x - x0f
        wy0, wy1 = y1f - y, y - y0f
        x0 = x0f.to(tl.int64)
        x1 = x1f.to(tl.int64)
        y0 = y0f.to(tl.int64)
        y1 = y1f.to(tl.int64)
        for c in range(C):
            base = ((bv * K + k) * C + c) * H * W
            v00 = tl.load(
                tgt_ptr + base + y0 * W + x0,
                mask=wmask & (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H),
                other=0.0,
            )
            v01 = tl.load(
                tgt_ptr + base + y1 * W + x0,
                mask=wmask & (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H),
                other=0.0,
            )
            v10 = tl.load(
                tgt_ptr + base + y0 * W + x1,
                mask=wmask & (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H),
                other=0.0,
            )
            v11 = tl.load(
                tgt_ptr + base + y1 * W + x1,
                mask=wmask & (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H),
                other=0.0,
            )
            sampled = (
                v00 * wx0 * wy0 + v01 * wx0 * wy1 + v10 * wx1 * wy0 + v11 * wx1 * wy1
            )
            refv = tl.load(
                ref_ptr + (bv * C + c) * H * W + h * W + ws, mask=wmask, other=0.0
            )
            acc += refv * sampled
    tl.store(
        out_ptr + (bv * D + d) * H * W + h * W + ws,
        acc / (K * tl.sqrt(float(C))),
        mask=wmask,
    )


def cost_volume_triton(ref, tgt, grid):
    """ref [BV,C,H,W], tgt [BV*K,C,H,W], grid [BV*K,D,H*W,2] -> [BV,D,H,W]."""
    bv, c, h, w = ref.shape
    bvk = tgt.shape[0]
    k = bvk // bv
    d = grid.shape[1] // h
    out = torch.empty(bv, d, h, w, device=ref.device, dtype=ref.dtype)
    cost_volume_kernel[(bv, d, h)](
        ref, tgt, grid, out, k, c, d, h, w, triton.next_power_of_2(w)
    )
    return out


if __name__ == "__main__":
    from zoo.model.resplat.unimatch.grid_sample_manual import grid_sample_manual

    torch.manual_seed(0)
    for bv, k, c, d, h, w in [(2, 2, 4, 8, 4, 6), (8, 2, 128, 128, 64, 120)]:
        ref = torch.randn(bv, c, h, w, device="cuda")
        tgt = torch.randn(bv * k, c, h, w, device="cuda")
        # include out-of-range and extreme coords: real grids carry
        # behind-camera rays with values far outside [-1, 1]
        grid = torch.rand(bv * k, d * h, w, 2, device="cuda") * 4 - 2
        grid.view(-1)[:7] = torch.tensor(
            [8.0, -9.0, 1e5, -1e5, 30.0, -30.0, 0.5], device="cuda"
        )
        got = cost_volume_triton(ref, tgt, grid)
        if bv * k * c * d * h * w < 2**24:  # reference fits in memory
            warped = grid_sample_manual(tgt, grid)
            want = (
                ref.reshape(bv, 1, c, 1, h, w) * warped.reshape(bv, k, c, d, h, w)
            ).sum(2) / (c**0.5)
            want = want.mean(1)
            print(
                f"BV={bv} C={c} D={d}: "
                f"max abs diff = {(got - want).abs().max().item():.3e}"
            )
        else:
            import time

            for i in range(2):
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                cost_volume_triton(ref, tgt, grid)
                torch.cuda.synchronize()
                print(
                    f"BV={bv} iter{i}: {time.perf_counter() - t0:.3f}s "
                    f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                )
