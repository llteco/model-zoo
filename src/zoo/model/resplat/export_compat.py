"""Export-compat patches for the vendored ReSplat encoder.

The TorchScript ONNX exporter (any opset) cannot lower two aten ops used by
the encoder; both patches are numerically equivalent pure-torch replacements:

  - ``aten::_cdist_forward`` (camera-neighbour distance matrix)
  - ``aten::linalg_inv_ex`` / ``aten::inverse`` (4x4/3x3 camera inverses;
    Gauss-Jordan with partial pivoting, static n so the loop unrolls)

Importing this module installs the patches (idempotent).
"""

import torch

_cdist_orig = torch.cdist
_installed = False


def _cdist_pure(a, b, p=2.0, **kwargs):
    if p == 2.0 and a.dim() == 3 and a.shape == b.shape:
        return (a.unsqueeze(3) - b.unsqueeze(2)).pow(2).sum(-1).sqrt()
    return _cdist_orig(a, b, p=p, **kwargs)


def _inverse_pure(a):
    """Batched inverse via Gauss-Jordan with partial pivoting."""
    n = int(a.shape[-1])
    batch = a.shape[:-2]
    eye = torch.eye(n, device=a.device, dtype=a.dtype).expand(*batch, n, n)
    aug = torch.cat([a, eye], dim=-1)
    rows = torch.arange(n, device=a.device)
    for i in range(n):
        mag = aug[..., i].abs().masked_fill(rows < i, float("-inf"))
        piv = mag.argmax(dim=-1)
        p_row = torch.gather(
            aug, -2, piv[..., None, None].expand(*batch, 1, 2 * n)
        ).squeeze(-2)
        i_row_old = aug[..., i, :]
        is_i = (rows == i).expand(*batch, n)
        is_p = rows.expand(*batch, n) == piv.unsqueeze(-1)
        swapped = torch.where(is_i.unsqueeze(-1), p_row.unsqueeze(-2), aug)
        aug = torch.where(is_p.unsqueeze(-1), i_row_old.unsqueeze(-2), swapped)
        piv_val = aug[..., i, i][..., None]
        row_i = aug[..., i, :] / (piv_val + 0.0)
        factors = aug[..., i][..., :, None]
        eliminated = aug - factors * row_i.unsqueeze(-2)
        aug = torch.where(
            is_i.unsqueeze(-1), row_i.unsqueeze(-2).expand_as(aug), eliminated
        )
    return aug[..., n:]


_diag_embed_orig = torch.diag_embed


def _diag_embed_pure(v, *args, **kwargs):
    """diag_embed(v) for the default (offset=0, last two dims) case."""
    if not args and not kwargs:
        n = int(v.shape[-1])
        eye = torch.eye(n, device=v.device, dtype=v.dtype)
        return v.unsqueeze(-1) * eye
    return _diag_embed_orig(v, *args, **kwargs)


_tensor_inverse_orig = torch.Tensor.inverse
_torch_inverse_orig = torch.inverse
_linalg_inv_orig = torch.linalg.inv


def install_export_compat() -> None:
    """Apply runtime patches (the TorchScript exporter does not run
    torch._decomp decompositions, so the ops must never appear in the
    traced graph at all)."""
    global _installed
    if _installed:
        return
    _installed = True
    torch.cdist = _cdist_pure
    torch.Tensor.inverse = lambda self: _inverse_pure(self)
    torch.inverse = _inverse_pure
    torch.linalg.inv = _inverse_pure
    torch.diag_embed = _diag_embed_pure
    torch.Tensor.diag_embed = lambda self: _diag_embed_pure(self)


@torch._decomp.register_decomposition(torch.ops.aten.inverse.default)
def _inverse_decomp(a):
    return _inverse_pure(a)


@torch._decomp.register_decomposition(torch.ops.aten.linalg_inv.default)
def _linalg_inv_decomp(a):
    return _inverse_pure(a)


@torch._decomp.register_decomposition(torch.ops.aten.linalg_inv_ex.default)
def _inv_ex_decomp(a, check_errors=False):
    return _inverse_pure(a), torch.zeros((), dtype=torch.int32, device=a.device)
