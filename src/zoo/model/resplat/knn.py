"""k-NN custom op for ReSplat (hyperonnx export + TensorRT plugin).

`KnnQuery` is an ``nn.Module`` so hyperonnx's hierarchical export
(``hier=[KnnQuery]``) turns its calls into a single ``hyper::KnnQuery`` ONNX
node, with the pure-torch implementation below inlined as the portable
function-body fallback. The zoo export post-process (see
``zoo/export/vision/resplat.py``) converts the node to ``trt::KnnQuery``
with an ``nsample`` attribute, implemented by the KnnQuery IPluginV3 plugin
in xpu-low-level (``src/nvidia/plugins/knn``).

Semantics follow pointops.knn_query: for each query point in ``new_xyz``,
the ``nsample`` nearest (squared euclidean) reference points within its
batch segment (``offset`` holds cumulative segment end indices). Returns
``int32 [M, nsample]`` indices ordered nearest-first.
"""

import torch
from torch import nn

# brute-force chunk size (rows of the [chunk, N] distance matrix)
_CHUNK = 2048


def knn_query_torch(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
    nsample: int,
) -> torch.Tensor:
    """Pure-torch kNN over point segments (export-friendly fallback)."""
    offset_l = offset.long()
    new_offset_l = new_offset.long()

    def seg_index(flat_idx: torch.Tensor, seg_ends: torch.Tensor) -> torch.Tensor:
        # segment id for each flat index (offsets are cumulative ends)
        return (flat_idx.unsqueeze(-1) >= seg_ends).sum(-1)

    device = xyz.device
    m_total = new_xyz.shape[0]
    flat = torch.arange(m_total, device=device)
    q_seg = seg_index(flat, new_offset_l)

    outs = []
    for s in range(int(new_offset_l.shape[0])):
        q_mask = q_seg == s
        qs = new_xyz[q_mask]
        ref = (
            xyz[: int(offset_l[s])]
            if s == 0
            else xyz[int(offset_l[s - 1]) : int(offset_l[s])]
        )
        k = min(nsample, int(ref.shape[0]))
        best_d = None
        best_i = None
        for c0 in range(0, int(qs.shape[0]), _CHUNK):
            q = qs[c0 : c0 + _CHUNK]
            d = torch.cdist(q.unsqueeze(0), ref.unsqueeze(0)).squeeze(0) ** 2
            kd = min(k, int(d.shape[1]))
            td, ti = torch.topk(d, kd, largest=False, dim=1)
            if kd < nsample:  # pad with -1 like pointops when segment is small
                pad_d = torch.full(
                    (td.shape[0], nsample - kd), float("inf"), device=device
                )
                pad_i = torch.full(
                    (ti.shape[0], nsample - kd), -1, dtype=ti.dtype, device=device
                )
                td = torch.cat([td, pad_d], dim=1)
                ti = torch.cat([ti, pad_i], dim=1)
            if best_d is None:
                best_d, best_i = td, ti
            else:
                best_d = torch.cat([best_d, td], dim=0)
                best_i = torch.cat([best_i, ti], dim=0)
        base = 0 if s == 0 else int(offset_l[s - 1])
        best_i = best_i + base
        if best_i.shape[0] == 0:
            best_i = torch.full((0, nsample), -1, dtype=torch.int32, device=device)
        outs.append((q_mask, best_i.to(torch.int32)))

    out = torch.empty(m_total, nsample, dtype=torch.int32, device=device)
    for q_mask, idx in outs:
        out[q_mask] = idx
    return out


def knn_query_and_group(
    feat: torch.Tensor,
    xyz: torch.Tensor,
    offset: torch.Tensor | None = None,
    new_xyz: torch.Tensor | None = None,
    new_offset: torch.Tensor | None = None,
    idx: torch.Tensor | None = None,
    nsample: int | None = None,
    with_xyz: bool = False,
):
    """Gather step of pointops.knn_query_and_group (pure torch; ``idx`` must
    be given, matching the model's usage where indices are precomputed)."""
    assert idx is not None and nsample is not None
    m = idx.shape[0]
    c = feat.shape[1]
    feat = torch.cat(
        [feat, torch.zeros(1, c, device=feat.device, dtype=feat.dtype)], dim=0
    )
    grouped = feat[idx.view(-1).long(), :].view(m, nsample, c)
    return grouped, idx


@torch.library.custom_op("resplat::knn_query_idx", mutates_args=())
def _knn_query_op(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
    nsample: int,
) -> torch.Tensor:
    """Opaque torch op wrapping the kNN (real impl: pure-torch chunked scan).

    The ONNX exporter emits it as a single ``trt::KnnQuery`` custom node
    (attribute ``nsample``), implemented by the KnnQuery IPluginV3 plugin in
    xpu-low-level ``src/nvidia/plugins/knn``."""
    return knn_query_torch(xyz, new_xyz, offset, new_offset, nsample)


@_knn_query_op.register_fake
def _(xyz, new_xyz, offset, new_offset, nsample):
    return xyz.new_empty((new_xyz.shape[0], nsample), dtype=torch.int32)


def _knn_query_symbolic(g, xyz, new_xyz, offset, new_offset, nsample):
    if not isinstance(nsample, int):
        nsample = torch.onnx.symbolic_helper._parse_arg(nsample, "i")
    return g.op("trt::KnnQuery", xyz, new_xyz, offset, new_offset, nsample_i=nsample)


torch.onnx.register_custom_op_symbolic("resplat::knn_query_idx", _knn_query_symbolic, 9)


class KnnQuery(nn.Module):
    """kNN as a module: exports as a single ``trt::KnnQuery`` plugin node
    (via the opaque ``resplat::knn_query_idx`` op), runs the pure-torch
    chunked implementation at eager/compile time."""

    def __init__(self, nsample: int):
        super().__init__()
        self.nsample = int(nsample)

    def forward(
        self,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        offset: torch.Tensor,
        new_offset: torch.Tensor,
    ) -> torch.Tensor:
        return _knn_query_op(xyz, new_xyz, offset, new_offset, self.nsample)
