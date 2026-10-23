"""Gaussian render + PLY helpers for the ReSplat serving backend.

Consolidates the former tests/resplat modules:

* ``fast_rasterize`` — thin torch adapter over the xpu-low-level splat
  renderer (``xpu.compiler.SplatRenderer`` — a port of stellaporta's
  streamlined gsplat renderer living in xpu-low-level ``src/nvidia/splat``,
  built as ``//src/nvidia/splat:splat_ffi`` and speaking the tvm FFI). The
  renderer consumes the gaussians as device pointers (DLPack zero-copy);
  the only per-view host work here is the quaternion repack and the
  intrinsics denormalization. ``render_view_gs`` mirrors the signature of
  the pure-torch reference below, except that the EWA projection consumes
  scales + rotations directly (what the engines emit), so no world-space
  covariance needs to be materialized. Color decode is INRA-style
  clamp(x + 0.5) over the SH output, which for degree 0 is exactly the
  reference renderer's (sh_dc * C0 + 0.5).clamp(0, 1).
* ``rasterize`` — the pure-torch tile-based EWA reference renderer (fp32,
  forward only); the automatic fallback when the splat FFI extension is
  unavailable and a cross-check for the fast path.
* ``ply_io`` — standard 3DGS binary PLY writer for the final gaussians.
"""
# pylint: disable=invalid-name,import-outside-toplevel,import-error

from functools import lru_cache

import numpy as np
import torch

# ---- fast path (xpu splat FFI) ----------------------------------------------


def build_covariance(
    scales: torch.Tensor, rotations_xyzw: torch.Tensor
) -> torch.Tensor:
    """World-space covariance from activated scales and xyzw rotations.

    Verbatim semantics of zoo.model.resplat.gaussians.build_covariance
    (pytorch3d's quaternion_to_matrix with the scipy xyzw unbind order).
    """

    i, j, k, r = torch.unbind(rotations_xyzw, dim=-1)
    two_s = 2.0 / ((rotations_xyzw * rotations_xyzw).sum(dim=-1) + 1e-8)
    rot = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    ).reshape(*rotations_xyzw.shape[:-1], 3, 3)
    scale = scales.diag_embed()
    return rot @ scale @ scale.transpose(-1, -2) @ rot.transpose(-1, -2)


@lru_cache(maxsize=1)
def renderer():
    """Process-wide SplatRenderer (lazily loads the splat_ffi library)."""
    from xpu.compiler import SplatRenderer

    return SplatRenderer()


def _sh_coeffs(
    harmonics: torch.Tensor, sh_degree: int | None
) -> tuple[torch.Tensor, int]:
    """harmonics [N, 3] (DC) or [N, 3, K] -> gsplat coeffs [N, K', 3] + degree.

    K' is capped at the requested degree's coefficient count so a degree-0
    render never copies the full 16-coefficient tensor.
    """

    if harmonics.dim() == 2:
        return harmonics.unsqueeze(1).contiguous(), 0  # [N, 3] DC -> [N, 1, 3]
    k = harmonics.shape[-1]
    degree = round(k**0.5) - 1
    assert (degree + 1) ** 2 == k, f"harmonics last dim {k} is not (deg+1)^2"
    if sh_degree is not None:
        degree = min(int(sh_degree), degree)
    coeffs = harmonics[..., : (degree + 1) ** 2]  # last dim is K
    return coeffs.transpose(1, 2).contiguous(), degree


@torch.no_grad()
def render_view_gs(
    means: torch.Tensor,  # [N, 3] world
    scales: torch.Tensor,  # [N, 3] activated
    rotations_xyzw: torch.Tensor,  # [N, 4] unit
    harmonics: torch.Tensor,  # [N, 3] DC or [N, 3, K] full
    opacities: torch.Tensor,  # [N] in (0, 1)
    c2w: torch.Tensor,  # [4, 4]
    intrinsics_norm: torch.Tensor,  # [3, 3] normalized by (W, H)
    height: int,
    width: int,
    bg_color: float = 0.0,
    sh_degree: int | None = None,  # cap below the harmonics' band count
    near_plane: float = 0.05,  # matches render_view's z > 0.05 cull
    far_plane: float = 1e10,
    return_alpha: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Render one view -> RGB [3, H, W] in [0, 1] (+ alpha [H, W] if asked)."""

    n = means.shape[0]
    if n == 0:
        rgb = torch.full((3, height, width), float(bg_color), device=means.device)
        return (
            (rgb, torch.zeros(height, width, device=means.device))
            if return_alpha
            else rgb
        )

    means = means.contiguous()
    scales = scales.contiguous()
    quats = torch.roll(rotations_xyzw, 1, dims=-1).contiguous()  # xyzw -> wxyz
    opacities = opacities.contiguous()

    w2c = torch.linalg.inv(c2w).contiguous()
    K = intrinsics_norm.clone()
    K[0, 0] *= width
    K[0, 2] *= width
    K[1, 1] *= height
    K[1, 2] *= height
    K = K.contiguous()

    coeffs, degree = _sh_coeffs(harmonics, sh_degree)

    # the FFI returns (rgb, alpha[, radii]) depending on its own flags
    result = renderer().render_view(
        means,
        quats,
        scales,
        opacities,
        coeffs,
        w2c,
        K,
        width=width,
        height=height,
        background=(float(bg_color),) * 3,
        sh_degree=degree,
        sh_degree_cap=degree,
        color_decode="inria",
        near_plane=near_plane,
        far_plane=far_plane,
    )
    rgb_hwc, alpha = result[0], result[1]
    rgb = rgb_hwc.permute(2, 0, 1).clamp(0.0, 1.0)  # HWC -> CHW
    return (rgb, alpha) if return_alpha else rgb


@torch.no_grad()
def render_views_gs(
    means: torch.Tensor,  # [N, 3]
    scales: torch.Tensor,  # [N, 3]
    rotations_xyzw: torch.Tensor,  # [N, 4]
    harmonics: torch.Tensor,  # [N, 3] / [N, 3, K]
    opacities: torch.Tensor,  # [N]
    extrinsics: torch.Tensor,  # [V, 4, 4] c2w
    intrinsics: torch.Tensor,  # [V, 3, 3] normalized
    height: int,
    width: int,
    bg_color: float = 0.0,
    sh_degree: int | None = None,
) -> torch.Tensor:
    """Render every context view -> [1, V, 3, H, W] (pipeline feedback)."""

    frames: list[torch.Tensor] = []
    for v in range(extrinsics.shape[0]):
        frame = render_view_gs(
            means,
            scales,
            rotations_xyzw,
            harmonics,
            opacities,
            extrinsics[v],
            intrinsics[v],
            height,
            width,
            bg_color=bg_color,
            sh_degree=sh_degree,
        )
        assert isinstance(frame, torch.Tensor)  # return_alpha defaults off
        frames.append(frame)
    return torch.stack(frames, dim=0).unsqueeze(0)


# ---- pure-torch reference (fallback + artifact renders) ---------------------

TILE = 16
CUTOFF_SQ = 9.0  # 3-sigma^2 mahalanobis cutoff
BG_COLOR = 1.0  # white background (indoor scenes)
SH_C0 = 0.28209479177387814


@torch.no_grad()
def render_view(
    means: torch.Tensor,  # [G, 3] world
    covariances: torch.Tensor,  # [G, 3, 3] world
    sh_dc: torch.Tensor,  # [G, 3] DC spherical-harmonic term
    opacities: torch.Tensor,  # [G] in (0, 1)
    c2w: torch.Tensor,  # [4, 4]
    intrinsics_norm: torch.Tensor,  # [3, 3] normalized by (W, H)
    height: int,
    width: int,
    bg_color: float = BG_COLOR,  # ReSplat trains renders on black (0.0)
) -> torch.Tensor:
    """Render one view -> RGB [3, H, W] in [0, 1].

    The standard tile-based EWA pipeline in fp32 torch: project means,
    per-tile assignment from 2D covariance radii, per-tile depth sort,
    front-to-back alpha compositing, SH DC term as the per-gaussian color.
    Not a production rasterizer: reference for the fast path and artifact
    generator for the render-vs-GT comparisons.

    Memory: per view the peak is a per-tile [G_tile, P] weight matrix (a
    few hundred rows), everything else is [G]-sized vectors.
    """
    device = means.device
    fx = intrinsics_norm[0, 0] * width
    fy = intrinsics_norm[1, 1] * height
    cx = intrinsics_norm[0, 2] * width
    cy = intrinsics_norm[1, 2] * height

    w2c = torch.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]

    p = means @ R.T + t
    z = p[:, 2]
    vis_idx = (z > 0.05).nonzero(as_tuple=True)[0]
    p, z = p[vis_idx], z[vis_idx]

    u = fx * p[:, 0] / z + cx
    v = fy * p[:, 1] / z + cy

    # EWA 2D covariance: Sigma' = (J R) Sigma (J R)^T over the xy-plane
    sig_cam = covariances[vis_idx]
    rc = sig_cam @ R.T  # rows in world -> camera axes
    zero = torch.zeros_like(z)
    j0 = torch.stack([fx / z, zero, -fx * p[:, 0] / (z * z)], dim=1)
    j1 = torch.stack([zero, fy / z, -fy * p[:, 1] / (z * z)], dim=1)
    r0 = (rc * j0[:, :, None]).sum(dim=1)  # [Gv, 3]
    r1 = (rc * j1[:, :, None]).sum(dim=1)
    var_u = (r0 * j0).sum(1) + 0.3  # +0.3 px low-pass (dilation)
    var_v = (r1 * j1).sum(1) + 0.3
    cov_uv = (r0 * j1).sum(1)

    det = (var_u * var_v - cov_uv * cov_uv).clamp_min(1e-9)
    inv_uu = var_v / det
    inv_vv = var_u / det
    inv_uv = -cov_uv / det

    lam = 0.5 * (
        var_u
        + var_v
        + torch.sqrt(((var_u - var_v) ** 2 + 4 * cov_uv * cov_uv).clamp_min(0))
    )
    radius_px = (3 * torch.sqrt(lam.clamp_min(1e-9))).ceil().clamp(1, 3 * TILE)

    # ---- per-tile assignment (vectorized over a small offset window) ----
    tiles_x = (width + TILE - 1) // TILE
    tiles_y = (height + TILE - 1) // TILE
    tx = (u / TILE).long()
    ty = (v / TILE).long()
    rad_t = (radius_px / TILE).ceil().long()
    max_r = int(rad_t.max().item())

    tile_ids_all, g_ids_all = [], []
    for dy in range(-max_r, max_r + 1):
        for dx in range(-max_r, max_r + 1):
            ttx, tty = tx + dx, ty + dy
            inside = (
                (ttx >= 0)
                & (ttx < tiles_x)
                & (tty >= 0)
                & (tty < tiles_y)
                & (abs(dx) <= rad_t)
                & (abs(dy) <= rad_t)
            )
            sel = inside.nonzero(as_tuple=True)[0]
            if sel.numel():
                tile_ids_all.append(tty[sel] * tiles_x + ttx[sel])
                g_ids_all.append(vis_idx[sel])
    tile_ids = torch.cat(tile_ids_all)
    g_ids = torch.cat(g_ids_all)

    order = torch.argsort(tile_ids, stable=True)
    tile_ids, g_ids = tile_ids[order], g_ids[order]
    boundaries = torch.searchsorted(
        tile_ids, torch.arange(1, tiles_x * tiles_y + 1, device=device)
    )

    # per-assignment precomputations (index into the visible subset)
    pos_in_vis = torch.searchsorted(vis_idx, g_ids)
    u_a, v_a = u[pos_in_vis], v[pos_in_vis]
    iuu, iuv, ivv = inv_uu[pos_in_vis], inv_uv[pos_in_vis], inv_vv[pos_in_vis]
    alpha_a = opacities[g_ids].clamp(1e-4, 0.99)
    color_a = (sh_dc[g_ids] * SH_C0 + 0.5).clamp(0, 1)
    depth_a = z[pos_in_vis]

    rgb = torch.empty(3, height, width, device=device)

    start = 0
    for t in range(tiles_x * tiles_y):
        end = int(boundaries[t])
        if end <= start:
            ty0, tx0 = t // tiles_x, t % tiles_x
            rgb[
                :,
                ty0 * TILE : ty0 * TILE + min(TILE, height - ty0 * TILE),
                tx0 * TILE : tx0 * TILE + min(TILE, width - tx0 * TILE),
            ] = bg_color
            start = end
            continue
        g = torch.arange(start, end, device=device)
        g = g[torch.argsort(depth_a[g])]  # front-to-back
        ty0, tx0 = t // tiles_x, t % tiles_x
        h_t = min(TILE, height - ty0 * TILE)
        w_t = min(TILE, width - tx0 * TILE)
        py = (ty0 * TILE + torch.arange(h_t, device=device)).float()
        px = (tx0 * TILE + torch.arange(w_t, device=device)).float()
        du = px[None, None, :] - u_a[g][:, None, None]  # [Gt, h, w]
        dv = py[None, :, None] - v_a[g][:, None, None]
        mahal = (
            iuu[g][:, None, None] * du * du
            + 2 * iuv[g][:, None, None] * du * dv
            + ivv[g][:, None, None] * dv * dv
        )
        a = torch.where(
            mahal <= CUTOFF_SQ,
            alpha_a[g][:, None, None] * torch.exp(-0.5 * mahal.clamp(min=0)),
            torch.zeros_like(mahal),
        )
        t_acc = torch.cumprod((1 - a).clamp(min=1e-6), dim=0)
        t_acc = torch.cat([torch.ones_like(t_acc[:1]), t_acc[:-1]], dim=0)
        w = a * t_acc  # [Gt, h, w]
        rgb_tile = torch.einsum("g h w, g c -> c h w", w, color_a[g])
        alpha_tile = w.sum(0).clamp(0, 1)
        rgb[:, ty0 * TILE : ty0 * TILE + h_t, tx0 * TILE : tx0 * TILE + w_t] = (
            rgb_tile + bg_color * (1 - alpha_tile[None])
        )
        start = end

    return rgb.clamp(0, 1)


# ---- 3DGS PLY serialization --------------------------------------------------


def write_gaussian_ply(path, means, scales, rotations_wxyz, sh_dc, sh_rest, opacities):
    """Write standard 3DGS binary PLY (inverse activations applied).

    means [N,3] | scales [N,3] (activated) | rotations_wxyz [N,4] (unit)
    sh_dc [N,3] | sh_rest [N,3,K] | opacities [N] in (0,1)
    """
    n = means.shape[0]
    n_rest = sh_rest.shape[1] * sh_rest.shape[2] if sh_rest is not None else 0

    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(3)]
    fields += [f"f_rest_{i}" for i in range(n_rest)]
    fields += ["opacity"]
    # INRIA 3DGS field order: rot BEFORE scale; the data block below has
    # always been written in this order, so any standards-compliant reader
    # (INRIA tools, stlp ply_loader) resolves the columns correctly.
    fields += [f"rot_{i}" for i in range(4)]
    fields += [f"scale_{i}" for i in range(3)]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        *[f"property float {f}" for f in fields],
        "end_header",
    ]
    block = "\n".join(header) + "\n"

    eps = 1e-6
    p = np.clip(opacities, eps, 1 - eps)
    op = (np.log(p) - np.log1p(-p))[:, None]  # logit: inverse of sigmoid
    sc = np.log(np.clip(scales, eps, None))
    if sh_rest is not None:
        rest = sh_rest.reshape(n, -1)
    else:
        rest = np.zeros((n, 0), dtype=np.float32)

    cols = [
        means,  # xyz
        np.zeros((n, 3), np.float32),  # normals (unused)
        sh_dc,  # f_dc
        rest,  # f_rest
        op,  # opacity (logit)
        rotations_wxyz,  # wxyz (matches the INRIA rot-before-scale header)
        sc,  # scales (log)
    ]
    data = np.concatenate(cols, axis=1).astype(np.float32)

    with open(path, "wb") as f:
        f.write(block.encode("ascii"))
        f.write(data.tobytes())
