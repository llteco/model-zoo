from collections.abc import Callable, Sequence
from functools import partial, reduce
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# mat_impl 性能评估（variant=tiny, 736x1280, fp32, RTX 5070 Ti, torch 2.12.1+cu130）:
#   eager:         cuda 62.0ms < pytorch_1d 78.6ms < pytorch(2d) 107.8ms
#   torch.compile: pytorch_1d 18.7ms < pytorch(2d) 25.3ms < cuda 32.3ms
# 结论: 默认编译路径下 pytorch_1d 最快（inductor 可融合 gather/softmax 链，
# cuda 自定义算子为不透明边界反而阻碍融合），故固定使用 pytorch_1d

DEFAULT_CHECKPOINT = (
    Path(__file__).parent.parent.parent.parent.parent
    / "models/stereo/matchstereo_rt.pth"
)


def init_coords(ref):
    B, H, W, _ = ref.shape
    # fp32 arange + cast: legacy TorchScript ONNX export fails with
    # "tensor does not have a device" on fp16 arange with dynamic end
    coords = torch.meshgrid(
        torch.arange(H, device=ref.device).to(ref.dtype),
        torch.arange(W, device=ref.device).to(ref.dtype),
        indexing="ij",
    )
    coords = torch.stack(coords[::-1], dim=-1)
    return coords[None].repeat(B, 1, 1, 1)


class LayerNormGeneral(nn.Module):
    def __init__(
        self,
        affine_shape: int | tuple[int, ...],
        normalized_dim=(-1,),
        scale=True,
        bias=False,
        eps=1e-6,
    ):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(
            x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps
        )


def stem(in_chs, out_chs, stride=2, act_layer=nn.GELU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=stride, padding=1),
        nn.InstanceNorm2d(out_chs),
        act_layer(),
    )


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.pre_norm = LayerNormGeneral(in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.pre_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        return x.permute(0, 2, 3, 1).contiguous()


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SepConv(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act1_layer=nn.GELU,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=3,
        padding=1,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=False,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MetaFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        token_mixer: type[nn.Module] = nn.Identity,
        mlp: type[nn.Module] = Mlp,
        mlp_ratio=4,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        drop=0.0,
        res_scale_init_value=None,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, drop=drop)
        self.norm1 = norm_layer(dim)
        self.res_scale1 = (
            Scale(dim=dim, init_value=res_scale_init_value)
            if res_scale_init_value
            else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.res_scale2 = (
            Scale(dim=dim, init_value=res_scale_init_value)
            if res_scale_init_value
            else nn.Identity()
        )

    def forward(self, x):
        x = x + self.res_scale1(self.token_mixer(self.norm1(x)))
        x = x + self.res_scale2(self.mlp(self.norm2(x)))
        return x


class MetaFormer(nn.Module):
    def __init__(
        self,
        in_chans=3,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers: Sequence[type[nn.Module]] = [nn.Identity],
        mlp: type[nn.Module] = Mlp,
        mlp_ratio=4,
        norm_layer: Callable[[int], nn.Module] = partial(
            LayerNormWithoutBias, eps=1e-6
        ),
    ):
        super().__init__()
        self.dims = dims
        num_stage = len(depths)
        self.num_stage = num_stage

        down_dims = [in_chans] + dims
        downsample_layers = [stem] + [Downsampling] * (len(depths) - 1)
        self.downsample_layers = nn.ModuleList(
            [
                downsample_layers[i](down_dims[i], down_dims[i + 1])
                for i in range(num_stage)
            ]
        )
        res_scale_init_values = [None] * (len(depths) - 2) + [1.0, 1.0]

        self.stages = nn.ModuleList()
        for i in range(num_stage):
            stage = nn.ModuleList(
                [
                    MetaFormerBlock(
                        dim=dims[i],
                        token_mixer=token_mixers[i],
                        mlp=mlp,
                        mlp_ratio=mlp_ratio,
                        norm_layer=norm_layer,
                        res_scale_init_value=res_scale_init_values[i],
                    )
                    for _ in range(depths[i])
                ]
            )
            self.stages.append(stage)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            if i == 0:
                x = x.permute(0, 2, 3, 1).contiguous()
            for block in cast(nn.ModuleList, self.stages[i]):
                x = block(x)
            outs.append(x)
        return outs


def convformer():
    depths = [2, 2, 6, 2]
    return MetaFormer(
        depths=depths,
        dims=[32, 64, 128, 256],
        mlp=Mlp,
        mlp_ratio=2,
        token_mixers=[SepConv] * len(depths),
    )


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1.permute(0, 3, 1, 2).contiguous())
        x2 = x2.permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x).permute(0, 2, 3, 1).contiguous()


class ConvGLU(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=2,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=hidden_features,
        )
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = (
            self.act(
                self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
            )
            * v
        )
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalCorrelation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNormWithoutBias(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.scale = dim**-0.5
        self.ot_iter = 3

    def _sinkhorn(self, attn, log_mu, log_nu):
        v = log_nu - torch.logsumexp(attn, dim=2)
        u = log_mu - torch.logsumexp(attn + v.unsqueeze(2), dim=3)
        for _ in range(self.ot_iter - 1):
            v = log_nu - torch.logsumexp(attn + u.unsqueeze(3), dim=2)
            u = log_mu - torch.logsumexp(attn + v.unsqueeze(2), dim=3)
        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def _optimal_transport(self, attn):
        w = attn.shape[2]
        dtype = attn.dtype
        # no torch.tensor([w]) in the cat: it becomes a CPU constant during
        # TorchScript tracing and breaks legacy ONNX export's constant
        # folding with a cuda/cpu device mismatch
        marginal = torch.ones(w + 1, device=attn.device, dtype=dtype)
        marginal[-1] = w
        marginal = marginal / (2 * w)
        log_marginal = marginal.log().reshape(1, 1, w + 1)
        attn = F.pad(attn, (0, 1, 0, 1), "constant", 0)
        attn = self._sinkhorn(attn, log_marginal, log_marginal)
        w_tensor = torch.tensor(w, dtype=dtype, device=attn.device)
        log_const = torch.log(2 * w_tensor)
        return (attn[:, :, :-1, :-1] + log_const).exp()

    def forward(self, x):
        x = self.norm(x)
        ref, tgt = x.chunk(2, dim=0)
        ref, tgt = self.q(ref), self.k(tgt)
        B, H, W, _ = ref.shape
        correlation = torch.matmul(ref, tgt.transpose(-2, -1)) * self.scale
        keep = torch.tril(torch.ones(W, W, dtype=torch.bool, device=ref.device))
        # ponytail: a python float here becomes a float32 ONNX constant under
        # export and forces a fp16->fp32 cast in Where; keep it dtype-matched.
        fill_value = torch.tensor(
            -1e9 if correlation.dtype == torch.float32 else -1e4,
            dtype=correlation.dtype,
            device=correlation.device,
        )
        correlation = torch.cat(
            (
                correlation.masked_fill(~keep, fill_value),
                correlation.permute(0, 1, 3, 2).masked_fill(~keep.T, fill_value),
            ),
            dim=0,
        )
        correlation = self._optimal_transport(correlation)
        return cal_disp(correlation, W)


def disp_to_flow_for_fast_init(disp):
    # ponytail: upstream negated the first half-batch twice (net no-op); only the
    # zero y-channel append remains
    return torch.cat((disp, torch.zeros_like(disp)), dim=-1).contiguous()


def cal_disp(correlation, w):
    # ponytail: legacy ONNX export emits linspace in float32 regardless of the
    # requested fp16 dtype; build it fp32 then cast so the final chain stays
    # fp16 instead of upcasting via x_grid - correspondence_left.
    x_grid = torch.linspace(
        0, w - 1, w, device=correlation.device, dtype=torch.float32
    ).to(correlation.dtype)
    prob_max_ind = correlation.max(dim=-1)[1].unsqueeze(3)
    prob_l = 2
    masked_prob_pad = F.pad(correlation, (prob_l, prob_l), "constant", 0)
    offsets = range(2 * prob_l + 1)
    weights = [
        torch.gather(masked_prob_pad, index=prob_max_ind + idx, dim=-1)
        for idx in offsets
    ]
    conf = reduce(torch.add, weights)
    correspondence_left = reduce(
        torch.add,
        (
            weight * (prob_max_ind + idx - prob_l)
            for idx, weight in zip(offsets, weights)
        ),
    )
    # ponytail: 1e-4 as a python float becomes a float32 ONNX constant under
    # export, upcasting the fp16 Div result to float32; keep it dtype-matched.
    eps = torch.tensor(1e-4, dtype=correlation.dtype, device=correlation.device)
    correspondence_left = (correspondence_left + eps) / (conf + eps)
    disparity = -(x_grid.reshape(1, 1, w) - correspondence_left.squeeze(3)).unsqueeze(1)
    return disp_to_flow_for_fast_init(disparity.permute(0, 2, 3, 1))


def compute_linear_weights(grid):
    x = grid[..., 0]
    dx = x - torch.floor(x)
    return torch.stack([1 - dx, dx], dim=-1)


def compute_match_attention_1d(q, k, m_id, win_r, H, W):
    B, N, h, C = q.shape
    M = 2 * win_r + 2
    dx = torch.arange(-win_r, win_r + 2, device=q.device, dtype=torch.long)
    dy = torch.zeros_like(dx)
    offsets = torch.stack((dx, dy), dim=-1).reshape(M, 2)
    coords = m_id.unsqueeze(3) + offsets.view(1, 1, 1, M, 2)
    x_coords = coords[..., 0].clamp(0, W - 1)
    y_coords = coords[..., 1].clamp(0, H - 1)
    indices = y_coords * W + x_coords
    k_expanded = k.unsqueeze(3).expand(-1, -1, -1, M, -1)
    indices_gather = indices.unsqueeze(-1).expand(-1, -1, -1, -1, C)
    k_sampled = torch.gather(k_expanded, dim=1, index=indices_gather)
    output = -torch.abs(q.unsqueeze(3) - k_sampled).sum(dim=-1)
    return output, indices_gather


def attn_scatter_1d(attn, win_r):
    B, N, h, M = attn.shape
    attn_2d = attn.view(B, N, h, 2 * win_r + 2)
    win_left = attn_2d[..., : 2 * win_r + 1]
    win_right = attn_2d[..., 1 : 2 * win_r + 2]
    return torch.stack([win_left, win_right], dim=3)


def attn_gather_1d(attn_sub, win_r):
    B, N, h, _, M_sub = attn_sub.shape
    merged = attn_sub.new_zeros(B, N, h, 2 * win_r + 2)
    merged[..., : 2 * win_r + 1] += attn_sub[:, :, :, 0, :]
    merged[..., 1 : 2 * win_r + 2] += attn_sub[:, :, :, 1, :]
    return merged


def compute_linear_softmax(attn, bilinear_weight, win_r):
    attn_sub = attn_scatter_1d(attn, win_r)
    attn_weighted = bilinear_weight.unsqueeze(-1) * attn_sub.softmax(dim=-1)
    return attn_gather_1d(attn_weighted, win_r)


def attention_aggregate_1d(v, attn, indices_gather, win_r):
    B, N, h, C = v.shape
    M = 2 * win_r + 2
    v_expanded = v.unsqueeze(3).expand(-1, -1, -1, M, -1)
    v_sampled = torch.gather(v_expanded, dim=1, index=indices_gather)
    output = (attn.unsqueeze(-1) * v_sampled).sum(dim=3)
    return output.view(B, N, -1)


class MatchAttention(nn.Module):
    def __init__(
        self,
        dim,
        win_r=1,
        num_head=8,
        head_dim=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        proj_bias=False,
        cross=False,
    ):
        super().__init__()
        self.num_head = num_head
        self.cross = cross
        self.head_dim = dim // num_head if head_dim is None else head_dim
        self.scale = self.head_dim**-0.5
        self.attention_dim = self.num_head * self.head_dim
        self.win_r = win_r
        self.attn_num = 2 * win_r + 2

        self.q = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        # Fused GEMM operands, (re)built after any state_dict load. In
        # forward, projections with the same input run as one wide mm
        # (+split views) — numerically identical per output element.
        self.register_buffer("_fused_w", None, persistent=False)
        self.register_buffer("_fused_b", None, persistent=False)
        self.register_load_state_dict_post_hook(MatchAttention._rebuild_fused)
        self._rebuild_fused()
        self.attn_drop = nn.Dropout(attn_drop)
        if self.cross:
            self.g = nn.Sequential(
                nn.Linear(dim, self.attention_dim, bias=qkv_bias), nn.SiLU()
            )
            self.proj = nn.Linear(
                self.attention_dim + self.num_head * self.attn_num, dim, bias=proj_bias
            )
        else:
            self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def clamp_max_offset(self, max_offset, H, W):
        max_offset_x, max_offset_y = max_offset.chunk(2, dim=-1)
        max_offset_x = max_offset_x.clamp(min=self.win_r, max=W - 1 - self.win_r - 1e-3)
        max_offset_y = max_offset_y.clamp(min=self.win_r, max=H - 1 - self.win_r - 1e-3)
        return torch.cat((max_offset_x, max_offset_y), dim=-1).contiguous()

    def _rebuild_fused(self, *args):
        """Materialize fused projection weights from the live q/k/v params.

        self-attention: one [3C, D] operand for q|k|v (all read the same
        input); cross: a [2C, D] operand for k|v (q reads a different one).
        """
        with torch.no_grad():
            if self.cross:
                self._fused_w = torch.cat((self.k.weight, self.v.weight), 0)
                self._fused_b = (
                    None
                    if self.k.bias is None
                    else torch.cat((self.k.bias, self.v.bias), 0)
                )
            else:
                self._fused_w = torch.cat(
                    (self.q.weight, self.k.weight, self.v.weight), 0
                )
                self._fused_b = (
                    None
                    if self.q.bias is None
                    else torch.cat((self.q.bias, self.k.bias, self.v.bias), 0)
                )

    def _project_qkv(self, ref, tgt):
        if self.cross:
            q = F.linear(ref, self.q.weight, self.q.bias)
            kv = F.linear(tgt, self._fused_w, self._fused_b)
            k, v = kv.split(self.attention_dim, dim=-1)
            return q, k, v
        qkv = F.linear(ref, self._fused_w, self._fused_b)
        q, k, v = qkv.split(self.attention_dim, dim=-1)
        return q, k, v

    def forward(self, x, max_offset):
        B, H, W, _ = x.shape
        N = H * W
        assert 2 * self.win_r + 2 <= W
        x = x.view(B, N, -1).contiguous()
        if self.cross:
            ref_, tgt_ = x.chunk(2, dim=0)
            ref = torch.cat((ref_, tgt_), dim=0)
            tgt = torch.cat((tgt_, ref_), dim=0)
        else:
            ref, tgt = x, x
        q, k, v = self._project_qkv(ref, tgt)

        max_offset = self.clamp_max_offset(max_offset, H, W)
        m_id = torch.floor(max_offset).to(torch.long)
        bilinear_weight = compute_linear_weights(max_offset)
        attn, indices_gather = compute_match_attention_1d(
            q.view(B, N, self.num_head, -1),
            k.view(B, N, self.num_head, -1),
            m_id,
            self.win_r,
            H,
            W,
        )
        attn = attn * self.scale
        attn = compute_linear_softmax(attn, bilinear_weight, self.win_r)
        attn = self.attn_drop(attn)
        x = attention_aggregate_1d(
            v.view(B, N, self.num_head, -1), attn, indices_gather, self.win_r
        )

        if self.cross:
            x = self.g(ref) * x
            attn = attn.view(B, N, -1).contiguous()
            x = torch.cat((x, attn), dim=-1).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.view(B, H, W, -1).contiguous()


class MatchAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        win_r,
        num_head=8,
        head_dim=32,
        mlp: type[nn.Module] = ConvGLU,
        mlp_ratio=2,
        field_dim=2,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        drop=0.0,
    ):
        super().__init__()
        self.num_head = num_head
        self.field_dim = field_dim
        self.match_attention_self = MatchAttention(
            dim + self.field_dim + self.num_head * 2,
            win_r,
            num_head=num_head,
            head_dim=head_dim,
        )
        self.norm0 = norm_layer(dim + self.field_dim + self.num_head * 2)
        self.match_attention_cross = MatchAttention(
            dim + self.field_dim,
            win_r,
            num_head=num_head,
            head_dim=head_dim,
            cross=True,
        )
        self.norm1 = norm_layer(dim + self.field_dim)
        self.mlp = mlp(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.norm2 = norm_layer(dim)
        self.field_scale = nn.Parameter(0.1 * torch.ones(1, 1, 1, 2))

    def forward(self, x, self_rpos, field):
        B, H, W, _ = x.shape
        x = torch.cat(
            (x, field * self.field_scale.to(field.dtype), self_rpos), dim=-1
        ).contiguous()
        coords_0 = init_coords(field).repeat(1, 1, 1, self.num_head)
        self_offset = self_rpos + coords_0
        self_offset = self_offset.view(B, H * W, self.num_head, 2).contiguous()

        x = x + self.match_attention_self(self.norm0(x), self_offset)

        self_rpos = x[..., -(self.num_head * 2) :].contiguous()
        x = x[..., : -(self.num_head * 2)].contiguous()

        x[..., -1] = 0
        field = x[..., -self.field_dim :].contiguous() / self.field_scale.to(
            field.dtype
        )

        offset = field.repeat(1, 1, 1, self.num_head).contiguous() + coords_0
        offset = offset.view(B, H * W, self.num_head, 2).contiguous()

        x = x + self.match_attention_cross(self.norm1(x), offset)

        x[..., -1] = 0
        field = x[..., -self.field_dim :].contiguous() / self.field_scale.to(
            field.dtype
        )

        x = x[..., : -self.field_dim].contiguous()
        x = x + self.mlp(self.norm2(x))
        return x, self_rpos, field


class MatchAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        win_r=2,
        num_layer=6,
        num_head=8,
        head_dim=32,
        mlp=ConvGLU,
        mlp_ratio=2,
        field_dim=2,
        norm_layer=LayerNormWithoutBias,
        drop=0.0,
    ):
        super().__init__()
        self.num_head = num_head
        self.layers = nn.ModuleList(
            [
                MatchAttentionLayer(
                    dim,
                    win_r=win_r,
                    num_head=num_head,
                    head_dim=head_dim,
                    mlp=mlp,
                    mlp_ratio=mlp_ratio,
                    field_dim=field_dim,
                    norm_layer=norm_layer,
                    drop=drop,
                )
                for _ in range(num_layer)
            ]
        )

    def forward(self, x, self_rpos, field):
        B, H, W, _ = x.shape
        self_rpos = self_rpos.repeat(1, 1, 1, self.num_head)
        for layer in self.layers:
            x, self_rpos, field = layer(x, self_rpos, field)
        self_rpos = self_rpos.view(B, H, W, self.num_head, 2).mean(
            dim=-2, keepdim=False
        )
        return x, self_rpos, field


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_head=8,
        head_dim=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        proj_bias=False,
        cross=False,
    ):
        super().__init__()
        self.num_head = num_head
        self.cross = cross
        self.head_dim = dim // num_head if head_dim is None else head_dim
        self.scale = self.head_dim**-0.5
        self.attention_dim = self.num_head * self.head_dim
        self.q = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, _ = x.shape
        N = H * W
        if self.cross:
            ref_, tgt_ = x.chunk(2, dim=0)
            ref = torch.cat((ref_, tgt_), dim=0)
            tgt = torch.cat((tgt_, ref_), dim=0)
        else:
            ref, tgt = x, x
        q, k, v = (
            t.reshape(B, N, self.num_head, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            for t in (self.q(ref), self.k(tgt), self.v(tgt))
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_head=8,
        head_dim=32,
        mlp: type[nn.Module] = ConvGLU,
        mlp_ratio=2,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        drop=0.0,
    ):
        super().__init__()
        self.num_head = num_head
        self.self_attention = Attention(dim, num_head=num_head, head_dim=head_dim)
        self.cross_attention = Attention(
            dim, num_head=num_head, head_dim=head_dim, cross=True
        )
        self.mlp = mlp(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.self_attention(self.norm0(x))
        x = x + self.cross_attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_layer=2,
        num_head=4,
        head_dim=32,
        mlp=ConvGLU,
        mlp_ratio=2,
        norm_layer=LayerNormWithoutBias,
        drop=0.0,
    ):
        super().__init__()
        self.num_head = num_head
        self.layers = nn.ModuleList(
            [
                AttentionLayer(
                    dim,
                    num_head=num_head,
                    head_dim=head_dim,
                    mlp=mlp,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    drop=drop,
                )
                for _ in range(num_layer)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MatchStereo(nn.Module):
    def __init__(
        self,
        refine_win_rs=[1, 1, 1, 1],
        refine_nums=[8, 8, 8, 2],
        num_heads=[4, 4, 4, 4],
        mlp_ratios=[2, 2, 2, 2],
        checkpoint: str | None = str(DEFAULT_CHECKPOINT),
    ):
        super().__init__()
        self.encoder = convformer()
        self.channels = self.encoder.dims[::-1]
        self.refine_win_rs = refine_win_rs
        self.refine_nums = refine_nums
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.head_dims = [c // 4 // h for c, h in zip(self.channels, self.num_heads)]

        self.factor = 2
        self.factor_last = 4
        self.field_dim = 2

        self.up_decoders = nn.ModuleList()
        self.up_masks = nn.ModuleList()
        for i in range(len(self.channels)):
            if i > 0:
                self.up_decoders.append(UpConv(self.channels[i - 1], self.channels[i]))
                self.up_masks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.channels[i - 1], self.channels[i - 1], 3, padding=1
                        ),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            self.channels[i - 1], (self.factor**2) * 9, 1, padding=0
                        ),
                    )
                )
            else:
                self.up_decoders.append(nn.Identity())
                self.up_masks.append(nn.Identity())

        self.up_masks.append(
            nn.Sequential(
                nn.Conv2d(self.channels[-1], self.channels[-1] * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.channels[-1] * 2, (self.factor_last**2) * 9, 1, padding=0
                ),
            )
        )

        self.match_attentions = nn.ModuleList()
        for i in range(len(self.refine_nums)):
            self.match_attentions.append(
                MatchAttentionBlock(
                    self.channels[i],
                    win_r=self.refine_win_rs[i],
                    num_layer=self.refine_nums[i],
                    num_head=self.num_heads[i],
                    head_dim=self.head_dims[i],
                    mlp_ratio=self.mlp_ratios[i],
                    field_dim=self.field_dim,
                )
            )

        self.global_attention = AttentionBlock(
            self.channels[0],
            num_layer=2,
            num_head=self.num_heads[0],
            head_dim=self.head_dims[0],
            mlp_ratio=self.mlp_ratios[0],
        )
        self.init_correlation_volume = GlobalCorrelation(self.channels[0])

        self.apply(self._init_weights)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.load_state_dict(state["model"], strict=False)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def upsample_field(self, field, mask, factor):
        B, H, W, D = field.shape
        field = field.permute(0, 3, 1, 2)
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(field * factor, (3, 3), padding=1)
        up_flow = up_flow.view(B, D, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 4, 2, 5, 3, 1)
        return up_flow.reshape(B, factor * H, factor * W, D).contiguous()

    def forward(self, img0, img1):
        img0 = (2 * (img0 / 255.0) - 1.0).contiguous()
        img1 = (2 * (img1 / 255.0) - 1.0).contiguous()
        x = torch.cat((img0, img1), dim=0)
        features = list(self.encoder(x))[::-1]

        features[0] = self.global_attention(features[0])
        field = self.init_correlation_volume(features[0])
        self_rpos = torch.zeros_like(field)
        features[0], self_rpos, field = self.match_attentions[0](
            features[0], self_rpos, field
        )
        for i in range(1, len(features)):
            features[i] = self.up_decoders[i](features[i - 1], features[i])
            up_mask = self.up_masks[i](features[i - 1].permute(0, 3, 1, 2))
            self_rpos = self.upsample_field(self_rpos, up_mask, self.factor)
            field = self.upsample_field(field, up_mask, self.factor)
            features[i], self_rpos, field = self.match_attentions[i](
                features[i], self_rpos, field
            )

        return self.upsample_field(
            field, self.up_masks[-1](features[-1].permute(0, 3, 1, 2)), self.factor_last
        )
