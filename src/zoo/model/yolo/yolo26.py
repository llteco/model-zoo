"""Inference-only YOLO26n detector, ported from Ultralytics 8.4.215.

Trimmed port of ``ultralytics/nn`` (Conv/C3k2/C2PSA/SPPF/Detect) plus the
``yolo26.yaml`` graph for the ``n`` scale, so the zoo can export YOLO26 without
depending on the ``ultralytics`` package (which also pulls opencv, pandas, ...).
Only the NMS-free one-to-one head and the post-processed 300 detections are
kept; the one-to-many training branch and the loss are dropped.

Reference (AGPL-3.0): https://github.com/ultralytics/ultralytics/blob/v8.4.215
Weights: yolo26n.pt from the v8.4.0 assets release; ``zoo.export`` downloads it
(pooch) and caches the extracted state dict.
"""

import math
import os

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Pad for 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x: float, divisor: int = 8) -> int:
    """Nearest channel count divisible by `divisor`, as Ultralytics rounds it."""
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


class Conv(nn.Module):
    """Convolution + BatchNorm + SiLU."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Concat(nn.Module):
    """Concatenate the given tensors along `dimension`."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Bottleneck(nn.Module):
    """Standard bottleneck with an optional residual."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP bottleneck with three convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(
                Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3 with k x k bottlenecks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C2f(nn.Module):
    """Faster CSP bottleneck with two convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2(C2f):
    """C2f whose blocks are Bottleneck, C3k, or Bottleneck + PSABlock."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class SPPF(nn.Module):
    """Spatial pyramid pooling - fast (three chained 5x5 max pools)."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        # Square pooling only: the upstream CPU-only separable variant exists to
        # speed up backprop and is numerically equivalent at inference.
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class Attention(nn.Module):
    """Position-sensitive self-attention over a feature map."""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        q, k, v = (
            self.qkv(x)
            .view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n)
            .split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        )
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(
            v.reshape(b, c, h, w)
        )
        return self.proj(x)


class PSABlock(nn.Module):
    """Attention block followed by a feed-forward block, both residual."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    """CSP block stacking PSABlocks on half of the channels."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
                for _ in range(n)
            )
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    """Left/top/right/bottom distances to xyxy boxes."""
    lt, rb = distance.chunk(2, 1)
    return torch.cat((anchor_points - lt, anchor_points + rb), 1)


class Detect26(nn.Module):
    """YOLO26 NMS-free head: decode one-to-one outputs and keep the top-k boxes.

    Output is ``(B, max_det, 6)`` with ``[x1, y1, x2, y2, score, class]`` in
    input-image pixels, i.e. what Ultralytics' export path produces.
    """

    def __init__(self, nc: int = 80, ch=(64, 128, 256), max_det: int = 300):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 1  # yolo26.yaml: no DFL, boxes are plain distances
        self.max_det = max_det
        # Strides of the P3/P4/P5 levels, tied to the yolo26 head wiring.
        self.stride = (8, 16, 32)
        c2 = max(16, ch[0] // 4, 4 * self.reg_max)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )

    def _anchors(self, feats):
        """Cell-centre anchors (2, A) and per-anchor strides (1, A)."""
        anchors, strides = [], []
        for i, f in enumerate(feats):
            # Static shapes: the zoo exports one fixed input size per run.
            h, w = int(f.shape[2]), int(f.shape[3])
            sx = torch.arange(w, dtype=f.dtype, device=f.device) + 0.5
            sy = torch.arange(h, dtype=f.dtype, device=f.device) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchors.append(torch.stack((sx, sy), -1).view(-1, 2))
            strides.append(f.new_full((h * w, 1), self.stride[i]))
        return torch.cat(anchors).transpose(0, 1), torch.cat(strides).transpose(0, 1)

    def forward(self, x):
        b = x[0].shape[0]
        boxes = torch.cat(
            [self.cv2[i](x[i]).view(b, 4, -1) for i in range(self.nl)], dim=-1
        )
        scores = torch.cat(
            [self.cv3[i](x[i]).view(b, self.nc, -1) for i in range(self.nl)], -1
        )
        anchors, strides = self._anchors(x)
        # (B, A, ...) so that the anchor gather is a plain ONNX GatherElements
        # on one tensor; gathering the concatenated boxes+scores would need the
        # index to match every dimension.
        dbox = (dist2bbox(boxes, anchors.unsqueeze(0)) * strides).permute(0, 2, 1)
        scores = scores.sigmoid().permute(0, 2, 1)
        k = min(self.max_det, scores.shape[1] * self.nc)
        conf, idx = scores.flatten(1).topk(k, dim=1)
        picks = dbox.gather(1, (idx // self.nc)[..., None].expand(-1, -1, 4))
        labels = (idx % self.nc).to(conf.dtype)[..., None]
        return torch.cat((picks, conf[..., None], labels), -1)


class YOLO26(nn.Module):
    """YOLO26 detector for the `n` scale (width 0.25, depth 0.50).

    Layer indices follow ``yolo26.yaml``: head layers read backbone outputs by
    index (6 = P4/16, 4 = P3/8, 10 = SPPF+C2PSA tail) and upsample from the
    previous layer, so the graph is written out explicitly here.
    """

    def __init__(self, nc: int = 80, checkpoint: str | os.PathLike | None = None):
        super().__init__()

        # yolo26n scale: width 0.25 (max 1024 channels), depth 0.50 (2 -> 1 repeat)
        def c(x):
            return make_divisible(min(x, 1024) * 0.25, 8)

        c1_, c2_, c3_, c4_, c5_ = c(64), c(128), c(256), c(512), c(1024)
        self.model = nn.Sequential(
            Conv(3, c1_, 3, 2),  # 0 P1/2
            Conv(c1_, c2_, 3, 2),  # 1 P2/4
            C3k2(c2_, c3_, 1, False, 0.25),  # 2
            Conv(c3_, c3_, 3, 2),  # 3 P3/8
            C3k2(c3_, c4_, 1, False, 0.25),  # 4
            Conv(c4_, c4_, 3, 2),  # 5 P4/16
            C3k2(c4_, c4_, 1, True),  # 6
            Conv(c4_, c5_, 3, 2),  # 7 P5/32
            C3k2(c5_, c5_, 1, True),  # 8
            SPPF(c5_, c5_, 5, 3, True),  # 9
            C2PSA(c5_, c5_, 1),  # 10
            nn.Upsample(None, 2, "nearest"),  # 11
            Concat(1),  # 12
            C3k2(c5_ + c4_, c4_, 1, True),  # 13
            nn.Upsample(None, 2, "nearest"),  # 14
            Concat(1),  # 15
            C3k2(2 * c4_, c3_, 1, True),  # 16 P3/8-small
            Conv(c3_, c3_, 3, 2),  # 17
            Concat(1),  # 18
            C3k2(c3_ + c4_, c4_, 1, True),  # 19 P4/16-medium
            Conv(c4_, c4_, 3, 2),  # 20
            Concat(1),  # 21
            C3k2(c4_ + c5_, c5_, 1, True, 0.5, True),  # 22 P5/32-large
            Detect26(nc, (c3_, c4_, c5_)),  # 23
        )
        if checkpoint:
            self.load(checkpoint)

    def load(self, checkpoint: str | os.PathLike) -> None:
        """Load an Ultralytics checkpoint (fp16 weights in, fp32 model)."""
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        head = "model.23."
        target = self.state_dict()
        weights = {}
        for k, v in state.items():
            if k.startswith(head):
                if not k.startswith(f"{head}one2one_"):
                    continue  # one-to-many head is training-only
                k = k.replace(f"{head}one2one_", head, 1)
            if k in target:  # Ultralytics keeps conv biases, this port uses bias=False
                weights[k] = v.float() if v.is_floating_point() else v
        self.load_state_dict(weights)

    def forward(self, img):
        m = self.model
        x0 = m[0](img)
        x1 = m[1](x0)
        x2 = m[2](x1)
        x3 = m[3](x2)
        x4 = m[4](x3)
        x5 = m[5](x4)
        x6 = m[6](x5)
        x7 = m[7](x6)
        x8 = m[8](x7)
        x9 = m[9](x8)
        x10 = m[10](x9)
        h = m[12]([m[11](x10), x6])  # cat backbone P4
        x13 = m[13](h)
        h = m[15]([m[14](x13), x4])  # cat backbone P3
        x16 = m[16](h)
        h = m[18]([m[17](x16), x13])  # cat head P4
        x19 = m[19](h)
        h = m[21]([m[20](x19), x10])  # cat head P5
        x22 = m[22](h)
        return m[23]([x16, x19, x22])
