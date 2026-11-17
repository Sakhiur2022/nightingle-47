"""Models = feature extractor + classifier, so that the FedUP filter can sit between them.
LeNet-5 (MNIST), ResNet-18 / ResNet-34 (torchvision, 32x32 input, optional ImageNet init as in the official code),
a SimpleViT-style Transformer (CIFAR-10), TinyBERT (AG News), plus a 3-block ConvNet (FuGuard's model) and a tiny CNN
for the smoke test. The filter is  Linear(d, out_dim) -> ReLU -> Linear(out_dim, d) -> ReLU  (no biases), as in the
official `Adapter`."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FedModel(nn.Module):
    def __init__(self, extractor, classifier, feat_dim):
        super().__init__()
        self.extractor, self.classifier, self.feat_dim = extractor, classifier, feat_dim

    def features(self, x):
        return self.extractor(x)

    def forward(self, x):
        return self.classifier(self.extractor(x))


class Filter(nn.Module):
    def __init__(self, dim, out_dim=32, out_relu=True):
        super().__init__()
        self.enc = nn.Linear(dim, out_dim, bias=False)
        self.dec = nn.Linear(out_dim, dim, bias=False)
        self.out_relu = out_relu

    def forward(self, x):
        y = self.dec(F.relu(self.enc(x)))
        return F.relu(y) if self.out_relu else y

    def nbytes(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())


class FilteredModel(nn.Module):
    """M_E (+) M_Fi (+) M_cl : the original model with the plug-in filter between extractor and classifier.
    Removing the filter (use `base`) restores the original model: unlearning is reversible."""
    def __init__(self, base, filt):
        super().__init__()
        self.base, self.filt = base, filt

    def forward(self, x):
        with torch.no_grad():
            f = self.base.features(x)
        return self.base.classifier(self.filt(f.float()))


# ------------------------------------------------------------------------------------------------ extractors
class LeNetExtractor(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(in_ch, 6, 5), nn.Conv2d(6, 16, 5)
        self.fc3, self.fc2 = nn.Linear(256, 120), nn.Linear(120, 512)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.max_pool2d(self.conv2(x), 2, 2)          # no ReLU after conv2, as in the official LeNet
        x = F.relu(self.fc3(x.flatten(1)))
        return F.relu(self.fc2(x))                      # 512-d features (input of the classifier)


def _resnet(name, pretrained):
    import torchvision
    from torchvision.models import resnet18, resnet34
    try:
        from torchvision.models import ResNet18_Weights, ResNet34_Weights
        w = {"resnet18": ResNet18_Weights.IMAGENET1K_V1, "resnet34": ResNet34_Weights.IMAGENET1K_V1}[name] if pretrained else None
        m = {"resnet18": resnet18, "resnet34": resnet34}[name](weights=w)
    except Exception as e:  # noqa: BLE001
        print(f"[models] could not load ImageNet weights ({e!r}); using random initialisation")
        m = {"resnet18": resnet18, "resnet34": resnet34}[name](weights=None)
    m.fc = nn.Identity()
    return m, 512


def posemb_sincos_2d(h, w, dim, temperature=10000.0):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None]
    x = x.flatten()[:, None] * omega[None]
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)


class ViTExtractor(nn.Module):
    """SimpleViT-style: patch 4, dim 512, depth 6, 8 heads, mlp 512, sin-cos positions, mean pooling (CIFAR-10 sizes)."""
    def __init__(self, image_size=32, patch=4, dim=512, depth=6, heads=8, mlp_dim=512, in_ch=3):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dim, patch, patch)
        self.register_buffer("pos", posemb_sincos_2d(image_size // patch, image_size // patch, dim), persistent=False)
        layer = nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        return self.norm(self.enc(t).mean(1))


class BertExtractor(nn.Module):
    """x: int64 [B, 2, L] = (input_ids, attention_mask). Feature = pooled [CLS] output."""
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(name)
        self.dim = self.bert.config.hidden_size

    def forward(self, x):
        out = self.bert(input_ids=x[:, 0], attention_mask=x[:, 1])
        p = getattr(out, "pooler_output", None)
        return p if p is not None else out.last_hidden_state[:, 0]


def _convnet(in_ch=3, width=128, depth=3, img=32):
    """FuGuard's ConvNet: 3 x [conv3x3(128) - GroupNorm - ReLU - AvgPool2]."""
    layers, c = [], in_ch
    for _ in range(depth):
        layers += [nn.Conv2d(c, width, 3, padding=1), nn.GroupNorm(4, width), nn.ReLU(inplace=True), nn.AvgPool2d(2, 2)]
        c, img = width, img // 2
    return nn.Sequential(*layers, nn.Flatten()), width * img * img


def _tiny(in_ch=3):
    return nn.Sequential(nn.Conv2d(in_ch, 8, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(8, 16, 3, stride=2, padding=1),
                         nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(64, 64), nn.ReLU()), 64


def build_model(cfg):
    m, K = cfg["model"], cfg["num_classes"]
    if m == "lenet5":
        ext, d = LeNetExtractor(cfg.get("in_ch", 1)), 512
    elif m in ("resnet18", "resnet34"):
        ext, d = _resnet(m, cfg["pretrained"])
    elif m == "vit":
        ext, d = ViTExtractor(in_ch=cfg.get("in_ch", 3)), 512
    elif m == "tinybert":
        ext = BertExtractor(cfg["text_model"])
        d = ext.dim
    elif m == "convnet":
        ext, d = _convnet(cfg.get("in_ch", 3))
    elif m == "tiny":
        ext, d = _tiny(cfg.get("in_ch", 3))
    else:
        raise KeyError(m)
    return FedModel(ext, nn.Linear(d, K), d)
