"""CenterNet-style marker-centre detector (approach A, docs/design/local-model.md).

ResNet-34 encoder (torchvision, BSD-3; ImageNet weights) + a U-Net decoder
back to stride 2, so markers a few pixels apart still land on different
heatmap cells. Four heads on the stride-2 map:

- heat   (1)  marker-centre heatmap (class-agnostic: the detection does not
              depend on how reliably a source labels the shape)
- offset (2)  sub-cell centre offset
- shape  (6)  marker shape class (circle/square/triangle/diamond/cross/other)
- embed  (E)  associative embedding: markers of one series pull together,
              different series push apart (CornerNet); this is what groups
              detections into series, learnt from colour and shape at once.

v2 (aux=True, docs/design/local-model.md「方式A v2」) adds two heads:

- size   (1)  log marker diameter in input pixels (CenterNet's wh, one value)
- region (2)  plot-area and legend logits per cell (CACHED-style context:
              peaks outside the plot or inside a legend are dropped)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

MARKER_CLASSES = ("circle", "square", "triangle", "diamond", "cross", "other")
STRIDE = 2


class _Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


def _head(cin: int, cout: int, bias: float = 0.0) -> nn.Sequential:
    h = nn.Sequential(
        nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, cout, 1)
    )
    nn.init.constant_(h[-1].bias, bias)
    return h


class MarkerNet(nn.Module):
    def __init__(self, embed_dim: int = 8, pretrained_path: str | None = None,
                 aux: bool = False):
        super().__init__()
        self.aux = aux
        r = torchvision.models.resnet34()
        if pretrained_path:
            r.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # /2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.up4 = _Up(512, 256, 256)  # -> /16
        self.up3 = _Up(256, 128, 128)  # -> /8
        self.up2 = _Up(128, 64, 96)  # -> /4
        self.up1 = _Up(96, 64, 64)  # -> /2
        self.heat = _head(64, 1, bias=-2.19)  # prior 0.1, as CenterNet
        self.offset = _head(64, 2)
        self.shape = _head(64, len(MARKER_CLASSES))
        self.embed = _head(64, embed_dim)
        if aux:
            self.size = _head(64, 1, bias=1.6)  # exp(1.6) ~ 5 px
            self.region = _head(64, 2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):  # img: B,3,H,W in [0,1]
        x = (img - self.mean) / self.std
        s1 = self.stem(x)
        s2 = self.l1(self.pool(s1))
        s3 = self.l2(s2)
        s4 = self.l3(s3)
        s5 = self.l4(s4)
        d = self.up4(s5, s4)
        d = self.up3(d, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        out = {
            "heat": self.heat(d),
            "offset": self.offset(d),
            "shape": self.shape(d),
            "embed": self.embed(d),
        }
        if self.aux:
            out["size"] = self.size(d)
            out["region"] = self.region(d)
        return out


def pick_device() -> str:
    """cuda, else Apple MPS, else cpu (RCB_DEVICE overrides)."""
    import os

    if os.environ.get("RCB_DEVICE"):
        return os.environ["RCB_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def region_loss(logits, target, known):
    """BCE per region channel, only on images whose region is known.
    target: B,2,H,W in {0,1}; known: B,2 in {0,1}."""
    w = known[:, :, None, None].expand_as(target)
    if w.sum() == 0:
        return logits.sum() * 0
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * w).sum() / w.sum()


def focal_loss(logits, target, alpha: float = 2.0, beta: float = 4.0, neg_weight=None):
    """CenterNet's penalty-reduced focal loss; target is a Gaussian-splatted
    heatmap with exact 1 at centres. neg_weight (B,) scales each image's
    background term (partially labelled real figures)."""
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p**alpha * (1 - target) ** beta * neg
    if neg_weight is not None:
        neg_loss = neg_loss * neg_weight.view(-1, 1, 1, 1)
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def embedding_loss(embed, batch_idx, ys, xs, series_ids, margin: float = 1.0):
    """Pull markers of one series to their mean, push series means of one
    image apart by `margin` (CornerNet's associative embedding, L2 form)."""
    if len(batch_idx) == 0:
        return embed.sum() * 0, embed.sum() * 0
    e = embed[batch_idx, :, ys, xs].float()  # N,E
    pull, push, n_pull, n_push = e.new_zeros(()), e.new_zeros(()), 0, 0
    for b in batch_idx.unique():
        mb = batch_idx == b
        ids = series_ids[mb]
        eb = e[mb]
        means = []
        for sid in ids.unique():
            es = eb[ids == sid]
            m = es.mean(0)
            means.append(m)
            pull = pull + ((es - m) ** 2).sum(1).mean()
            n_pull += 1
        if len(means) > 1:
            M = torch.stack(means)
            dist = torch.cdist(M, M)
            k = len(means)
            mask = ~torch.eye(k, dtype=torch.bool, device=M.device)
            push = push + F.relu(margin - dist[mask]).pow(2).mean()
            n_push += 1
    return pull / max(n_pull, 1), push / max(n_push, 1)
