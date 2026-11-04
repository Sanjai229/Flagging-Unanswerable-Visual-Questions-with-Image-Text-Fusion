"""Every trainable model behind one interface.

A model is built from its config entry by build_model(spec, dims), declares the feature arrays it `needs`,
and maps a batch dict to one logit per pair (positive class = unanswerable). Late fusion has no trainable
weights, so it lives in train.py and combines saved predictions.
Fusion levels follow the early / late / intermediate taxonomy of Baltrusaitis, Ahuja & Morency (2019).
"""
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear, returning one logit."""

    def __init__(self, d_in, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPModel(nn.Module):
    """One input: a single-modality baseline. Several inputs: early fusion (vectors joined, then one MLP)."""

    def __init__(self, dims, inputs, hidden=256, dropout=0.3):
        super().__init__()
        self.needs = tuple(inputs)
        self.head = MLPHead(sum(dims[k] for k in inputs), hidden, dropout)

    def forward(self, b):
        return self.head(torch.cat([b[k] for k in self.needs], -1))


class CrossAttentionBlock(nn.Module):
    """Question tokens attend to image regions, then image regions attend to the updated question tokens.
    Scaled dot-product attention (Vaswani et al. 2017) used across modalities as in Tsai et al. (2019)."""

    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.q_from_v = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.v_from_q = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln_q, self.ln_v = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, q, q_mask, v):
        q = self.ln_q(q + self.q_from_v(q, v, v)[0])
        v = self.ln_v(v + self.v_from_q(v, q, q, key_padding_mask=~q_mask)[0])
        return q, v


class CrossAttentionFusion(nn.Module):
    """Intermediate fusion over question word tokens (BERT) and image region tokens (ViT, 7x7 grid)."""

    def __init__(self, dims, q_tokens="txt_tok", q_mask="txt_mask", v_tokens="img_tok",
                 d_model=256, heads=4, layers=1, dropout=0.1):
        super().__init__()
        self.keys = (q_tokens, q_mask, v_tokens)
        self.needs = self.keys
        self.q_proj, self.v_proj = nn.Linear(dims[q_tokens], d_model), nn.Linear(dims[v_tokens], d_model)
        self.blocks = nn.ModuleList(CrossAttentionBlock(d_model, heads, dropout) for _ in range(layers))
        self.head = MLPHead(2 * d_model, 128, 0.3)

    def forward(self, b):
        kq, km, kv = self.keys
        q, v, mask = self.q_proj(b[kq]), self.v_proj(b[kv]), b[km]
        for block in self.blocks:
            q, v = block(q, mask, v)
        m = mask.unsqueeze(-1).float()
        q_pool = (q * m).sum(1) / m.sum(1).clamp(min=1)       # mean over real (non-padding) words
        return self.head(torch.cat([q_pool, v.mean(1)], -1))


class GatedFusion(nn.Module):
    """Gated Multimodal Unit (Arevalo et al. 2017) with modality dropout.

    A learned gate z in (0, 1), per pair and hidden unit, weights the question representation by z and the photo
    representation by 1 - z. Modality dropout blanks the question (first half of the dropped share) or the photo
    (second half) for some training pairs, so the gate learns to cope when one input is unreliable.
    The mean gate per pair is kept in `last_gate_q` (higher = more weight on the question).
    """

    def __init__(self, dims, q_input="txt_mean", v_input="img_cls", hidden=256, dropout=0.3, modality_dropout=0.15):
        super().__init__()
        self.keys = (q_input, v_input)
        self.needs = self.keys
        self.h_q, self.h_v = nn.Linear(dims[q_input], hidden), nn.Linear(dims[v_input], hidden)
        self.gate = nn.Linear(dims[q_input] + dims[v_input], hidden)
        self.head = MLPHead(hidden, hidden, dropout)
        self.p = modality_dropout
        self.last_gate_q = None

    def forward(self, b):
        x_q, x_v = b[self.keys[0]], b[self.keys[1]]
        if self.training and self.p > 0:
            u = torch.rand(len(x_q), 1, device=x_q.device)
            x_q = x_q * (u >= self.p / 2)
            x_v = x_v * ((u < self.p / 2) | (u >= self.p))
        z = torch.sigmoid(self.gate(torch.cat([x_q, x_v], -1)))
        h = z * torch.tanh(self.h_q(x_q)) + (1 - z) * torch.tanh(self.h_v(x_v))
        self.last_gate_q = z.mean(-1).detach()
        return self.head(h)


MODEL_TYPES = {"mlp": MLPModel, "cross_attention": CrossAttentionFusion, "gated": GatedFusion}
NOT_HYPERPARAMETERS = {"type", "group", "label", "members"}


def build_model(spec, dims):
    """Build a model from its config entry; `dims` maps feature name -> vector size."""
    kind = spec["type"]
    if kind not in MODEL_TYPES:
        raise ValueError(f"unknown model type '{kind}'; choose from {sorted(MODEL_TYPES)} (late fusion: see train.py)")
    params = {k: v for k, v in spec.items() if k not in NOT_HYPERPARAMETERS}
    return MODEL_TYPES[kind](dims, **params)


def model_inputs(spec):
    """Feature arrays a config entry needs (without building the model)."""
    kind = spec["type"]
    if kind == "mlp":
        return list(spec["inputs"])
    if kind == "cross_attention":
        return [spec.get("q_tokens", "txt_tok"), spec.get("q_mask", "txt_mask"), spec.get("v_tokens", "img_tok")]
    if kind == "gated":
        return [spec.get("q_input", "txt_mean"), spec.get("v_input", "img_cls")]
    return []
