"""PolyRoom: Room-aware Transformer for Floorplan Reconstruction (ECCV 2024).

Ported from https://github.com/3dv-casia/PolyRoom (``models/roomformer.py``,
``models/deformable_transformer.py``, ``models/backbone.py`` and
``models/position_encoding.py``) so that the network can be exported to ONNX.

The port keeps every parameter name of the reference implementation, so the
official checkpoints still load.  Only execution-level details changed:

* ``MultiScaleDeformableAttention`` (a compiled CUDA extension) is replaced by
  an equivalent ``F.grid_sample`` implementation, which exports to ONNX
  ``GridSample``.
* ``NestedTensor`` padding masks are gone: the models are exported for a single
  unpadded image, so every mask is known to be all-valid.
* The room-query initialisation is a padded ``(coords, mask)`` pair instead of a
  Python list of per-image polygons, which makes it a graph input.
* ``nn.MultiheadAttention`` is evaluated with an explicit (mathematically
  identical) attention implementation instead of its CUDA fast path, which has
  no ONNX symbolic.

Coordinates are normalised to ``[0, 1]`` by 255, matching the 256x256
top-down density maps used by the reference data loader (``poly_data.py``) and
``util.poly_ops.pad_gt_polys``.
"""

import copy
import math
from typing import cast

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen statistics, as used by the reference backbone."""

    weight: Tensor
    bias: Tensor
    running_mean: Tensor
    running_var: Tensor

    def __init__(self, n: int, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class PositionEmbeddingSine(nn.Module):
    """Sine position embedding, identical to the reference implementation."""

    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(x.dtype)


def _all_valid(feat: Tensor) -> Tensor:
    """Padding mask of a feature map; the export assumes a single unpadded image."""
    return torch.zeros(
        (feat.shape[0], feat.shape[2], feat.shape[3]),
        dtype=torch.bool,
        device=feat.device,
    )


class Backbone(nn.Module):
    """ResNet backbone returning ``layer2``/``layer3``/``layer4`` features."""

    def __init__(
        self,
        name: str = "resnet50",
        return_interm_layers: bool = True,
        dilation: bool = False,
    ):
        super().__init__()
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=None,
            norm_layer=FrozenBatchNorm2d,
        )
        # single-channel (top-down density) input, as in the reference model
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if name in ("resnet18", "resnet34"):
            raise ValueError("number of channels are hard coded")
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        return self.body(image)


class Joiner(nn.Sequential):
    """Backbone plus position embedding, returning features, masks and encodings."""

    def __init__(self, backbone: Backbone, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, input: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        xs = cast(Backbone, self[0])(input)
        pos_embed = cast(PositionEmbeddingSine, self[1])
        features = [xs[name] for name in sorted(xs)]
        masks = [_all_valid(feat) for feat in features]
        pos = [pos_embed(feat, mask) for feat, mask in zip(features, masks)]
        return features, masks, pos


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention with ``grid_sample`` based sampling.

    Semantics follow the reference CUDA kernel: locations are normalised to
    ``[0, 1]``, bilinear sampling uses ``align_corners=False`` and samples that
    fall outside a feature map contribute zero.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, got {d_model} and {n_heads}"
            )
        if d_model // n_heads & (d_model // n_heads - 1):
            raise ValueError("d_model // n_heads must be a power of two")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2
        )
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> Tensor:
        """Sample ``value`` at ``reference_points + offsets``.

        Args:
            query: ``(N, Lq, C)`` decoder/encoder queries.
            reference_points: ``(N, Lq, n_levels, 2)`` normalised anchor points.
            value: ``(N, sum(H*W), C)`` concatenated multi-level features.
            spatial_shapes: per-level ``(H, W)``, must match ``value``.
        """
        n, len_q, _ = query.shape
        n, len_in, _ = value.shape
        if reference_points.shape[-1] != 2:
            raise ValueError("only 2D reference points are supported")

        value = self.value_proj(value)
        value = value.reshape(n, len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).reshape(
            n, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attn = self.attention_weights(query).reshape(
            n, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attn = F.softmax(attn, -1).reshape(
            n, len_q, self.n_heads, self.n_levels, self.n_points
        )

        # offsets are predicted in normalised units, reference points use (x, y)
        offset_normalizer = torch.tensor(
            [(w, h) for h, w in spatial_shapes],
            dtype=query.dtype,
            device=query.device,
        )
        sampling_locations = reference_points[:, :, None, :, None, :] + (
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        output = self._sample(value, spatial_shapes, sampling_locations, attn)
        return self.output_proj(output)

    def _sample(
        self,
        value: Tensor,
        spatial_shapes: list[tuple[int, int]],
        sampling_locations: Tensor,
        attn: Tensor,
    ) -> Tensor:
        n, _, m, d = value.shape
        len_q = sampling_locations.shape[1]
        heads = value.permute(0, 2, 1, 3).reshape(n * m, -1, d)

        sampled = []
        for level, piece in enumerate(
            torch.split(heads, [h * w for h, w in spatial_shapes], dim=1)
        ):
            h, w = spatial_shapes[level]
            feat = piece.reshape(n, m, h * w, d).permute(0, 1, 3, 2)
            feat = feat.reshape(n * m, d, h, w)
            grid = sampling_locations[:, :, :, level]
            grid = 2 * grid - 1
            grid = grid.permute(0, 2, 1, 3, 4).reshape(n * m, len_q, self.n_points, 2)
            sampled.append(
                F.grid_sample(
                    feat,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )

        # (N, Lq, M, L, P) -> (N*M, 1, Lq, L*P) to match the sampled values
        weights = attn.permute(0, 2, 1, 3, 4).reshape(
            n * m, 1, len_q, self.n_levels * self.n_points
        )
        out = (torch.cat(sampled, dim=-1) * weights).sum(-1)
        return out.reshape(n, m * d, len_q).permute(0, 2, 1)


def _attention_no_weights(
    mha: nn.MultiheadAttention, query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """``nn.MultiheadAttention`` forward without its CUDA fast path or weights.

    Exportable equivalent of ``mha(query, key, value, need_weights=False)[0]``
    for 3D inputs and no mask.
    """
    length, batch, embed_dim = query.shape
    heads = mha.num_heads
    head_dim = embed_dim // heads
    weight = mha.in_proj_weight
    bias = mha.in_proj_bias
    q, k, v = F.linear(query, weight, bias).chunk(3, dim=-1)
    q = q.reshape(length, batch * heads, head_dim).transpose(0, 1)
    k = k.reshape(length, batch * heads, head_dim).transpose(0, 1)
    v = v.reshape(length, batch * heads, head_dim).transpose(0, 1)
    attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(head_dim), -1)
    out = torch.bmm(attn, v).transpose(0, 1).reshape(length, batch, embed_dim)
    return F.linear(out, mha.out_proj.weight, mha.out_proj.bias)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: Tensor) -> Tensor:
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        reference_points: Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> Tensor:
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, src, spatial_shapes
        )
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(
        spatial_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        device: torch.device,
    ) -> Tensor:
        reference_points_list = []
        for level, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * w)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        spatial_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        pos: Tensor,
    ) -> Tensor:
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.self_attn_1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # present in the released checkpoints but never used by the reference
        # forward; kept so the official state dict loads with strict=True
        self.norm2_1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: Tensor) -> Tensor:
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout4(tgt2))

    def forward(
        self,
        tgt: Tensor,
        query_pos: Tensor | None,
        reference_points: Tensor,
        src: Tensor,
        spatial_shapes: list[tuple[int, int]],
        num_polys: int,
        num_queries_per_poly: int,
    ) -> Tensor:
        q = k = self.with_pos_embed(tgt, query_pos)
        batch = q.shape[0]
        # in-room self attention
        tgt2 = _attention_no_weights(
            self.self_attn,
            q.reshape(-1, num_queries_per_poly, q.shape[-1]).transpose(0, 1),
            k.reshape(-1, num_queries_per_poly, q.shape[-1]).transpose(0, 1),
            tgt.reshape(-1, num_queries_per_poly, q.shape[-1]).transpose(0, 1),
        )
        tgt2 = tgt2.transpose(0, 1).reshape(batch, -1, q.shape[-1])
        # between-room self attention
        tgt2_2 = _attention_no_weights(
            self.self_attn_1,
            q.reshape(-1, num_polys, q.shape[-1]).transpose(0, 1),
            k.reshape(-1, num_polys, q.shape[-1]).transpose(0, 1),
            tgt.reshape(-1, num_polys, q.shape[-1]).transpose(0, 1),
        )
        tgt2_2 = tgt2_2.transpose(0, 1).reshape(batch, -1, q.shape[-1])
        tgt = self.norm2(tgt + self.dropout2(tgt2 + tgt2_2))

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos), reference_points, src, spatial_shapes
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        return self.forward_ffn(tgt)


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer: nn.Module, num_layers: int, query_pos_type: str = "none"
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_pos_type = query_pos_type
        self.coords_embed: nn.ModuleList | None = None
        self.class_embed: nn.ModuleList | None = None
        self.pos_trans: nn.Module | None = None
        self.pos_trans_norm: nn.Module | None = None

    @staticmethod
    def get_query_pos_embed(ref_points: Tensor) -> Tensor:
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=ref_points.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        ref_points = ref_points * scale
        pos = ref_points[:, :, :, None] / dim_t
        return torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)

    def forward(
        self,
        tgt: Tensor,
        reference_points: Tensor,
        src: Tensor,
        spatial_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        num_polys: int,
        num_queries_per_poly: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output = tgt
        intermediates = []
        intermediate_references = []
        intermediate_classes = []
        assert self.coords_embed is not None and self.class_embed is not None
        for level, layer in enumerate(self.layers):
            reference_points_input = (
                reference_points[:, :, None] * valid_ratios[:, None]
            )
            if self.query_pos_type == "sine":
                assert self.pos_trans is not None and self.pos_trans_norm is not None
                query_pos = self.pos_trans_norm(
                    self.pos_trans(self.get_query_pos_embed(reference_points))
                )
            else:
                query_pos = None
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                spatial_shapes,
                num_polys,
                num_queries_per_poly,
            )
            # iterative polygon refinement
            offset = cast(MLP, self.coords_embed[level])(output)
            reference_points = (offset + inverse_sigmoid(reference_points)).sigmoid()
            point_classes = cast(nn.Linear, self.class_embed[level])(output)
            intermediates.append(output)
            intermediate_references.append(reference_points)
            intermediate_classes.append(point_classes)
        return (
            torch.stack(intermediates),
            torch.stack(intermediate_references),
            torch.stack(intermediate_classes),
        )


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        enc_n_points: int = 4,
        query_pos_type: str = "sine",
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.query_pos_type = query_pos_type

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, query_pos_type
        )

        self.level_embed = nn.Parameter(torch.empty(num_feature_levels, d_model))

        if query_pos_type == "sine":
            self.decoder.pos_trans = nn.Linear(d_model, d_model)
            self.decoder.pos_trans_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(
        self,
        srcs: list[Tensor],
        masks: list[Tensor],
        pos_embeds: list[Tensor],
        query_embed: Tensor,
        tgt: Tensor,
        room_inial: Tensor | None,
        room_inial_mask: Tensor | None,
        num_polys: int,
        num_queries_per_poly: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            _, _, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            lvl_pos_embed = pos_embed.flatten(2).transpose(1, 2) + self.level_embed[
                level
            ].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        src_flatten = torch.cat(src_flatten, 1)
        pos_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, valid_ratios, pos_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        reference_points = query_embed.unsqueeze(0).expand(bs, -1, -1).sigmoid()
        if room_inial is not None and room_inial_mask is not None:
            keep = room_inial_mask[..., None] > 0
            reference_points = torch.where(keep, room_inial, reference_points)
        query_embed = inverse_sigmoid(reference_points)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        hs, inter_references, inter_classes = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            valid_ratios,
            num_polys,
            num_queries_per_poly,
        )
        return hs, reference_points, inter_references, inter_classes


class MLP(nn.Module):
    """Very simple multi-layer perceptron (FFN)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim], strict=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RoomFormer(nn.Module):
    """PolyRoom floorplan reconstruction network.

    ``forward`` returns ``(pred_logits, pred_coords)`` (plus ``pred_room_logits``
    when ``semantic_classes > 0``):

    * ``pred_logits``: ``(B, num_polys, num_queries_per_poly)`` corner validity
    * ``pred_coords``: ``(B, num_polys, num_queries_per_poly, 2)`` normalised
    * ``pred_room_logits``: ``(B, num_polys, semantic_classes)`` room classes
    """

    def __init__(
        self,
        backbone: Joiner,
        transformer: DeformableTransformer,
        num_classes: int = 1,
        num_queries: int = 800,
        num_polys: int = 20,
        num_feature_levels: int = 4,
        semantic_classes: int = -1,
    ):
        super().__init__()
        if num_queries % num_polys != 0:
            raise ValueError("num_queries must be divisible by num_polys")
        self.num_queries = num_queries
        self.num_polys = num_polys
        self.num_queries_per_poly = num_queries // num_polys
        self.num_feature_levels = num_feature_levels
        self.transformer = transformer
        hidden_dim = transformer.d_model

        class_head = nn.Linear(hidden_dim, num_classes)
        coords_head = MLP(hidden_dim, hidden_dim, 2, 3)

        self.query_embed = nn.Embedding(num_queries, 2)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            in_channels = backbone.num_channels[-1]
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = backbone.num_channels[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_head.bias.data = torch.ones(num_classes) * bias_value
        last_coords = cast(nn.Linear, cast(MLP, coords_head).layers[-1])
        nn.init.constant_(last_coords.weight.data, 0)
        nn.init.constant_(cast(Tensor, last_coords.bias).data, 0)
        for proj in self.input_proj:
            conv = cast(nn.Conv2d, cast(nn.Sequential, proj)[0])
            nn.init.xavier_uniform_(conv.weight, gain=1)
            nn.init.constant_(cast(Tensor, conv.bias), 0)

        # per-layer prediction heads (dense supervision, as in the reference model)
        num_pred = transformer.decoder.num_layers
        self.class_embed = _get_clones(class_head, num_pred)
        self.coords_embed = _get_clones(coords_head, num_pred)
        transformer.decoder.coords_embed = self.coords_embed
        transformer.decoder.class_embed = self.class_embed

        # semantically-rich floorplan (optional room classes)
        self.room_class_embed = (
            nn.Linear(hidden_dim, semantic_classes) if semantic_classes > 0 else None
        )
        # kept for checkpoint compatibility; not used by the forward pass
        self.room_label_embed = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        image: Tensor,
        room_inial: Tensor | None = None,
        room_inial_mask: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        features, masks, pos = self.backbone(image)
        bs = image.shape[0]
        srcs = []
        for level, feat in enumerate(features):
            srcs.append(self.input_proj[level](feat))
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for level in range(_len_srcs, self.num_feature_levels):
                if level == _len_srcs:
                    src = self.input_proj[level](features[-1])
                else:
                    src = self.input_proj[level](srcs[-1])
                mask = _all_valid(src)
                pos_embed = cast(PositionEmbeddingSine, self.backbone[1])
                pos_l = pos_embed(src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        hs, _, inter_references, inter_classes = self.transformer(
            srcs,
            masks,
            pos,
            self.query_embed.weight,
            self.tgt_embed.weight,
            room_inial,
            room_inial_mask,
            self.num_polys,
            self.num_queries_per_poly,
        )

        num_layer = hs.shape[0]
        outputs_class = inter_classes.reshape(
            num_layer, bs, self.num_polys, self.num_queries_per_poly
        )
        outputs_coord = inter_references.reshape(
            num_layer, bs, self.num_polys, self.num_queries_per_poly, 2
        )
        out: tuple[Tensor, ...] = (outputs_class[-1], outputs_coord[-1])
        if self.room_class_embed is not None:
            room_logits = self.room_class_embed(
                hs[-1].view(bs, self.num_polys, self.num_queries_per_poly, -1).mean(2)
            )
            out = out + (room_logits,)
        return out


def build_polyroom(
    num_queries: int = 800,
    num_polys: int = 20,
    num_feature_levels: int = 4,
    hidden_dim: int = 256,
    nheads: int = 8,
    enc_layers: int = 6,
    dec_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    enc_n_points: int = 4,
    dec_n_points: int = 4,
    query_pos_type: str = "sine",
    semantic_classes: int = -1,
    dilation: bool = False,
) -> RoomFormer:
    """Build PolyRoom with the hyper-parameters of the reference implementation."""
    position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone = Joiner(
        Backbone(return_interm_layers=num_feature_levels > 1, dilation=dilation),
        position_embedding,
    )
    transformer = DeformableTransformer(
        d_model=hidden_dim,
        nhead=nheads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="relu",
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        enc_n_points=enc_n_points,
        query_pos_type=query_pos_type,
    )
    return RoomFormer(
        backbone,
        transformer,
        num_classes=1,
        num_queries=num_queries,
        num_polys=num_polys,
        num_feature_levels=num_feature_levels,
        semantic_classes=semantic_classes,
    )


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse of the sigmoid, clamped exactly like the reference helper."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
