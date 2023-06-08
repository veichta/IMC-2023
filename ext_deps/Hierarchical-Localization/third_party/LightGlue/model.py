from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from torch import nn

try:
    from flash import flash_attn_kvpacked_func, flash_attn_qkvpacked_func
except ModuleNotFoundError:
    print("Module flash_attn not found. Disabling FlashAttention.")


def merge_dict(dict1, dict2, strict=True):
    dict1 = deepcopy(dict1)
    for key, val in dict2.items():
        if key in dict1 and type(val) == dict:
            assert isinstance(dict1[key], dict), key
            dict1[key] = merge_dict(dict1[key], val)
        elif key in dict1 or not strict:
            dict1[key] = val
        else:
            raise RuntimeError(f"Key {key} not found in dict1.")
    return dict1


class RecursiveNamespace(SimpleNamespace):
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

    def todict(self):
        d = deepcopy(vars(self))
        for k in d.keys():
            if isinstance(d[k], RecursiveNamespace):
                d[k] = d[k].todict()
        return d


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(kpts, size=None, shape=None):
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one * w, one * h])[None]
    size = size.to(kpts.device)
    shift = size.float() / 2
    scale = size.max(1).values.float() / 2  # actual SuperGlue mult by 0.7
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_cached_rotary_emb(freqs, t):
    if freqs.dim() >= t.dim():
        # rotate
        return (t * freqs[0]) + (rotate_half(t) * freqs[1])
    else:
        # fourier
        return t + freqs.unsqueeze(-2)


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim=None, gamma=1.0, H_dim=0):
        """
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf
        """
        super().__init__()
        self.M = M
        self.F_dim = F_dim if F_dim is not None else dim
        self.H_dim = H_dim
        self.D = dim
        self.gamma = gamma
        self.add_mlp = H_dim is not None and H_dim > 0

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        if self.add_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.F_dim, self.H_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.H_dim, self.D),
            )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x):
        B, N, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        if not self.add_mlp:
            emb = torch.stack([cosines, sines], 0).unsqueeze(-2)
            return repeat(emb, "... n -> ... (n r)", r=2)
        else:
            F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
            # Step 2. Compute projected Fourier features (eq. 6)
            return self.mlp(F)


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=0.01, learned_freq=False):
        super().__init__()
        freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        self.freqs_x = nn.Parameter(freqs_x, requires_grad=learned_freq)
        self.freqs_y = nn.Parameter(freqs_y, requires_grad=learned_freq)

    def forward(self, pos):
        freqs_x = torch.einsum(
            "..., f -> ... f", pos[..., 0].type(self.freqs_x.dtype), self.freqs_x
        )
        freqs_x = repeat(freqs_x, "... n -> ... (n r)", r=2)

        freqs_y = torch.einsum(
            "..., f -> ... f", pos[..., 1].type(self.freqs_y.dtype), self.freqs_y
        )
        freqs_y = repeat(freqs_y, "... n -> ... (n r)", r=2)

        freqs = torch.cat([freqs_x, freqs_y], dim=-1).unsqueeze(-2)
        return torch.stack([freqs.cos(), freqs.sin()], 0)


class TokenConfidence(nn.Module):
    def __init__(self, dim):
        super(TokenConfidence, self).__init__()
        self.dim = dim

        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def get_tokens(self, desc0, desc1):
        return self.token(desc0.detach().float()).squeeze(-1), self.token(
            desc1.detach().float()
        ).squeeze(-1)

    def forward(self, desc0, desc1):
        token0, token1 = self.get_tokens(desc0, desc1)
        return token0, token1

    def stop(self, desc0, desc1, conf_th, inl_th):
        assert desc0.shape[0] == 1
        token0, token1 = self.get_tokens(desc0, desc1)
        tokens = torch.cat([token0, token1], -1)
        if conf_th:
            pos = (tokens > conf_th).float().mean()
            return pos > inl_th
        else:
            return tokens.mean() > inl_th

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        correct1 = la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class FastAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, q, k, v, mask=None):
        b, n, h, d = q.shape
        query, key, value = [
            x.permute(2, 0, 1, 3).contiguous().view(b * h, -1, d) for x in [q, k, v]
        ]
        score = torch.bmm(query, key.transpose(1, 2)) * self.scale
        attn = torch.nn.functional.softmax(score, -1)
        context = torch.bmm(attn, value)
        context = context.view(h, b, -1, d)
        context = context.permute(1, 2, 0, 3)
        return context


class FlashAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

    def forward(self, q, k, v, weights=None):
        return flash_attn_qkvpacked_func(torch.stack([q, k, v], 2).half()).to(q.dtype)


class TorchFlashAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, q, k, v, weights=None):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=False
        ):
            args = [x.bfloat16().transpose(1, 2).contiguous() for x in [q, k, v]]
            return F.scaled_dot_product_attention(*args).to(q.dtype).transpose(1, 2)


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, flash=False, bias=True) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        attn = TorchFlashAttention if flash else FastAttention
        self.inner_attn = attn(self.head_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def _forward(self, x, encoding=None, mask=None):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b s (h d three) -> b s h d three", three=3, h=self.num_heads)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if encoding is not None:
            q = apply_cached_rotary_emb(encoding, q)
            k = apply_cached_rotary_emb(encoding, k)

        context = self.inner_attn(q, k, v)
        message = self.out_proj(rearrange(context, "b s h d -> b s (h d)"))
        return x + self.ffn(torch.cat([x, message], -1))

    def forward(self, x0, x1, encoding0=None, encoding1=None):
        return self._forward(x0, encoding0), self._forward(x1, encoding1)


def bisoftmax(sim, rel_pos_bias):
    sim = sim if rel_pos_bias is None else sim + rel_pos_bias
    return (
        nn.functional.softmax(sim, dim=-1),
        nn.functional.softmax(sim.transpose(-2, -1).contiguous(), dim=-1).transpose(-2, -1),
        1,
        1,
    )


class CrossTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        flash=False,  # breaks bidirectionality
        bias=True,
    ):
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)

        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

        if flash:
            self.flash = TorchFlashAttention(dim_head)
        else:
            self.flash = None

    def map_(self, func, x0, x1):
        return func(x0), func(x1)

    def forward(
        self,
        x0,
        x1,
        rel_pos_bias=None,
    ):
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (qk0, qk1, v0, v1)
        )
        if self.flash is not None:
            # @TODO: Add bidirectional flash attn kernel
            qk0, qk1, v0, v1 = [x.bfloat16().transpose(1, 2) for x in [qk0, qk1, v0, v1]]
            # m0 = self.flash(qk0, torch.stack([qk1, v1], 2))
            # m1 = self.flash(qk1, torch.stack([qk0, v0], 2))
            m0 = self.flash(qk0, qk1, v1)
            m1 = self.flash(qk1, qk0, v0)
            m0, m1 = [x.to(x0.dtype).transpose(1, 2) for x in [m0, m1]]
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("b h i d, b h j d -> b h i j", qk0, qk1)
            expsim0, expsim1, invlse0, invlse1 = bisoftmax(sim, rel_pos_bias)
            m0 = torch.einsum("bhij, bhjd -> bhid", expsim0, v1) * invlse0
            m1 = torch.einsum("bhji, bhjd -> bhid", expsim1, v0) * invlse1
        m0, m1 = self.map_(lambda t: rearrange(t, "b h n d -> b n (h d)"), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


def sigmoid_log_double_softmax(sim, z0, z1, r):
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = torch.nn.functional.log_softmax(sim, 2)
    # scores1 = torch.nn.functional.log_softmax(sim, 1)
    scores1 = torch.nn.functional.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(
        -1, -2
    )
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = (scores0 + scores1 + certainties) * r
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim, use_sqrt=False, tmp=None):
        super(MatchAssignment, self).__init__()
        self.use_sqrt = use_sqrt
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.temp = None if tmp is None else nn.Parameter(torch.Tensor([tmp]))
        r = torch.Tensor([0.5 if use_sqrt else 1.0])
        self.register_buffer("r", r)

    def forward(self, desc0, desc1):
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        if self.temp is None:
            _, _, d = mdesc0.shape
            mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
            sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        else:
            mdesc0 = F.normalize(mdesc0, dim=-1)
            mdesc1 = F.normalize(mdesc1, dim=-1)
            sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1) * self.temp

        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1, self.r / (1 + self.use_sqrt))
        return scores, sim

    def scores(self, desc0, desc1):
        m0 = torch.sigmoid(self.matchability(desc0)).squeeze(-1)
        m1 = torch.sigmoid(self.matchability(desc1)).squeeze(-1)
        return m0, m1


def filter_matches(scores, th, on_bins=False, zero_out=False):
    if zero_out:
        # Zero out rows / cols
        outliers0 = scores[:, :-1, -1].exp() > (1.0 - th)
        outliers1 = scores[:, -1, :-1].exp() > (1.0 - th)
        scores = scores.exp()
        scores[:, :-1, :-1] *= (1.0 - outliers0.unsqueeze(-1).float()) * (
            1.0 - outliers1.unsqueeze(1).float()
        )
        scores = scores.log()
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    mutual0 = torch.arange(m0.shape[1]).to(m0)[None] == m1.gather(1, m0)
    mutual1 = torch.arange(m1.shape[1]).to(m1)[None] == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    if on_bins:
        bin_scores0 = 1.0 - scores[:, :-1, -1].exp()
        bin_scores1 = 1.0 - scores[:, -1, :-1].exp()
        max0_exp = torch.min(bin_scores0, bin_scores1.gather(1, m0))
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    if th is not None:
        valid0 = mutual0 & (mscores0 > th)
    else:
        valid0 = mutual0
    # valid0 = mutual0 & (z0.squeeze(-1) > 0.8)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, m0.new_tensor(-1))
    m1 = torch.where(valid1, m1, m1.new_tensor(-1))
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,
        "descriptor_dim": 256,
        "rotary": {
            "do": True,
            "hidden_dim": None,
            "axial": False,
        },
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,
        "temp": None,
        "filter_threshold": 0.2,
        "filter_bins": False,
        "early_stop_th": -1,
        "adaptive_width": False,
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
        "use_sqrt": False,
        "checkpointed": True,
        "weights": None,
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]

    def __init__(self, conf):
        super().__init__()
        # default_conf = OmegaConf.create(self.default_conf)
        # OmegaConf.set_struct(default_conf, True)
        # conf = self.conf = OmegaConf.merge(default_conf, conf)
        # OmegaConf.set_readonly(conf, True)
        conf = self.conf = RecursiveNamespace(**merge_dict(self.default_conf, conf))
        print(conf.todict())
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        if conf.rotary.do:
            if not conf.rotary.axial:
                head_dim = conf.descriptor_dim // conf.num_heads
                self.posenc = LearnableFourierPositionalEncoding(
                    2, head_dim, head_dim, H_dim=conf.rotary.hidden_dim
                )
            else:
                self.posenc = AxialRotaryEmbedding(
                    conf.descriptor_dim // (conf.num_heads * 2), learned_freq=True, theta=0.01
                )
        else:
            self.keypoint_encoder = nn.Sequential(
                nn.Linear(4, 2 * conf.descriptor_dim),
                nn.LayerNorm(2 * conf.descriptor_dim),
                nn.ReLU(),
                nn.Linear(2 * conf.descriptor_dim, conf.descriptor_dim),
            )

        n = conf.n_layers
        d = conf.descriptor_dim
        h = conf.num_heads
        self.self_attn = nn.ModuleList([Transformer(d, h, conf.flash) for _ in range(n)])

        self.cross_attn = nn.ModuleList([CrossTransformer(d, h, conf.flash) for _ in range(n)])

        self.log_assignment = nn.ModuleList(
            [MatchAssignment(d, conf.use_sqrt, conf.temp) for _ in range(n)]
        )

        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])

        if conf.weights is not None:
            if conf.weights.endswith(".pth"):
                path = conf.weights
            else:
                path = Path(__file__).parent
                path = path / "weights/{}.pth".format(self.conf.weights)
            state_dict = torch.load(str(path), map_location="cpu")
            self.load_state_dict(state_dict)

    def forward(self, data):
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0_, kpts1_ = data["keypoints0"], data["keypoints1"]

        b, m, _ = kpts0_.shape
        b, n, _ = kpts1_.shape

        kpts0 = normalize_keypoints(
            kpts0_, size=data.get("image_size0"), shape=data["image0"].shape
        )
        kpts1 = normalize_keypoints(
            kpts1_, size=data.get("image_size1"), shape=data["image1"].shape
        )

        assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
        assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)

        desc0 = data["descriptors0"].detach().transpose(-1, -2)
        desc1 = data["descriptors1"].detach().transpose(-1, -2)

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        if self.conf.rotary.do:
            encoding0 = self.posenc(kpts0)
            encoding1 = self.posenc(kpts1)
            if self.training and self.conf.checkpointed:
                # dummy for checkpointing
                desc0 = desc0 + torch.zeros(
                    1, dtype=desc0.dtype, device=desc0.device, requires_grad=True
                )
                desc1 = desc1 + torch.zeros(
                    1, dtype=desc1.dtype, device=desc1.device, requires_grad=True
                )
        else:
            encoding0, encoding1 = None, None
            pos0 = torch.cat(
                [kpts0, data["oris0"][..., None].cos(), data["oris0"][..., None].sin()], -1
            ).float()
            pos1 = torch.cat(
                [kpts1, data["oris1"][..., None].cos(), data["oris1"][..., None].sin()], -1
            ).float()
            desc0 = desc0 + self.keypoint_encoder(pos0)
            desc1 = desc1 + self.keypoint_encoder(pos1)

        # GNN + final_proj + assignment
        all_desc0, all_desc1 = [], []
        ind0 = torch.arange(0, m).to(device=kpts0.device)[None]
        ind1 = torch.arange(0, n).to(device=kpts0.device)[None]
        avg_width = torch.zeros(self.conf.n_layers).to(kpts0)
        avg_width[0] = (kpts0.shape[1] + kpts1.shape[1]) / 2.0
        for i in range(self.conf.n_layers):
            if self.training and self.conf.checkpointed:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                    self.self_attn[i], desc0, desc1, encoding0, encoding1, preserve_rng_state=False
                )

                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                    self.cross_attn[i], desc0, desc1, preserve_rng_state=False
                )
            else:
                avg_width[i] = (desc0.shape[1] + desc1.shape[1]) / 2.0
                desc0, desc1 = self.self_attn[i](desc0, desc1, encoding0, encoding1)
                desc0, desc1 = self.cross_attn[i](desc0, desc1)

            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue

            if self.conf.early_stop_th > 0:
                if self.token_confidence[i].stop(
                    desc0, desc1, self.conf_th(i), self.conf.early_stop_th
                ):
                    all_desc0.append(desc0)
                    all_desc1.append(desc1)
                    break

            if self.conf.adaptive_width:
                assert desc0.shape[0] == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                # token0, token1 = None, None
                match0, match1 = self.log_assignment[i].scores(desc0, desc1)

                mask0 = self.get_mask(token0, match0, self.conf_th(i), 0.01)
                mask1 = self.get_mask(token1, match1, self.conf_th(i), 0.01)

                ind0 = ind0[mask0][None]
                ind1 = ind1[mask1][None]

                desc0 = desc0[mask0][None]
                desc1 = desc1[mask1][None]

                if encoding0 is not None:
                    encoding0 = encoding0[:, mask0][:, None]
                    encoding1 = encoding1[:, mask1][:, None]

        # scatter with indices
        if self.conf.adaptive_width:
            scores = -torch.ones((b, m + 1, n + 1), device=kpts0.device) * torch.inf
            scores_, _ = self.log_assignment[i](desc0, desc1)
            scores[:, ind0[0], -1] = scores_[:, :-1, -1]
            scores[:, -1, ind1[0]] = scores_[:, -1, :-1]
            x, y = torch.meshgrid(ind0[0], ind1[0], indexing="ij")
            scores[:, x, y] = scores_[:, :-1, :-1]
        else:
            scores, _ = self.log_assignment[i](desc0, desc1)

        m0, m1, mscores0, mscores1 = filter_matches(
            scores, self.conf.filter_threshold, on_bins=self.conf.filter_bins
        )

        return {
            "log_assignment": scores,
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "stop": torch.Tensor([i + 1]),
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
        }

    def conf_th(self, i):
        t = 4.0
        return np.clip(0.8 + 0.1 * np.exp(-t * (i - 1) / self.conf.n_layers), 0, 1)

    def get_mask(self, confidence, match, conf_th, match_th):
        if conf_th:
            mask = torch.where(confidence > conf_th, match, match.new_tensor(1.0)) > match_th
        else:
            mask = match > match_th
        return mask
