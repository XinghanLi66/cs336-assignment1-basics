import math

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn


class Linear(nn.Module):
    """Minimal linear layer without bias that matches nn.Linear's API subset."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, object] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        weight = torch.empty((out_features, in_features), **factory_kwargs)
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Simple embedding lookup layer mirroring nn.Embedding (without extras)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, object] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        weight = torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... d_model"]:
        token_ids = token_ids.to(dtype=torch.long)
        return self.weight[token_ids]


class SwiGLU(nn.Module):
    """Position-wise feed-forward network using the SwiGLU activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            target = (8.0 * d_model) / 3.0
            d_ff = max(64, 64 * math.ceil(target / 64))
        self.d_model = d_model
        self.d_ff = d_ff
        factory_kwargs: dict[str, object] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        x_w1 = self.w1(x)
        x_w3 = self.w3(x)
        gated = F.silu(x_w1) * x_w3
        return self.w2(gated)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) as per https://arxiv.org/abs/1910.07467."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, object] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        weight = torch.empty((d_model,), **factory_kwargs)
        nn.init.ones_(weight)
        self.weight = nn.Parameter(weight)
        self.eps = eps

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        cast_dtype = torch.float32 if in_dtype in (torch.bfloat16, torch.float16) else in_dtype
        x = x.to(dtype=cast_dtype)
        rms = torch.sqrt(torch.mean(torch.pow(x, 2), -1, keepdim=True) + self.eps)
        output = (x / rms) * self.weight
        return output.to(dtype=in_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Applies rotary positional embeddings (RoPE) to queries/keys."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("`d_k` must be even for RoPE.")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        device_arg = device
        index = torch.arange(0, d_k, 2, device=device_arg, dtype=dtype)
        freq_exponents = index / d_k
        frequencies = theta ** (-freq_exponents)

        position_kwargs: dict[str, object] = {}
        if device is not None:
            position_kwargs["device"] = device
        positions = torch.arange(max_seq_len, dtype=dtype, **position_kwargs).unsqueeze(-1)
        angles: Float[torch.Tensor, "max_seq_len d_k/2"] = positions * frequencies
        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        cos = torch.stack((cos_values, cos_values), dim=-1)
        sin = torch.stack((sin_values, sin_values), dim=-1)
        self.register_buffer("cos_cached", cos.reshape(max_seq_len, d_k), persistent=False)
        self.register_buffer("sin_cached", sin.reshape(max_seq_len, d_k), persistent=False)
        self.cos_cached: Float[torch.Tensor, "max_seq_len d_k"]
        self.sin_cached: Float[torch.Tensor, "max_seq_len d_k"]

    def forward(
        self,
        x: Float[torch.Tensor, "... seq d_k"],
        token_positions: Int[torch.Tensor, "... seq"],
    ) -> Float[torch.Tensor, "... seq d_k"]:
        in_dtype = x.dtype
        x = x.to(dtype=self.cos_cached.dtype)
        if token_positions.shape != x.shape[:-1]:
            token_positions = torch.broadcast_to(token_positions, x.shape[:-1])

        token_positions = token_positions.to(dtype=torch.long, device=x.device)

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        cos_even = cos[..., ::2]
        sin_even = sin[..., ::2]

        rotated_even = x_even * cos_even - x_odd * sin_even
        rotated_odd = x_even * sin_even + x_odd * cos_even

        rotated = torch.empty_like(x)
        rotated[..., ::2] = rotated_even
        rotated[..., 1::2] = rotated_odd
        return rotated.to(dtype=in_dtype)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically-stable softmax along the specified dimension."""

    original_dtype = x.dtype
    working = x.to(torch.float32) if original_dtype in {torch.float16, torch.bfloat16} else x
    shifted = working - torch.amax(working, dim=dim, keepdim=True)
    exponentiated = torch.exp(shifted)
    normalized = exponentiated / torch.sum(exponentiated, dim=dim, keepdim=True)
    return normalized.to(dtype=original_dtype)


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... query d_k"],
    K: Float[torch.Tensor, "... key d_k"],
    V: Float[torch.Tensor, "... key d_v"],
    mask: Bool[torch.Tensor, "... query key"] | None = None,
) -> Float[torch.Tensor, "... query d_v"]:
    d_k = Q.shape[-1]
    if d_k == 0:
        raise ValueError("Last dimension of Q (d_k) must be positive.")

    original_dtype = Q.dtype
    working_dtype = torch.float32 if original_dtype in {torch.float16, torch.bfloat16} else original_dtype

    Q_working = Q.to(working_dtype)
    K_working = K.to(working_dtype)
    V_working = V.to(working_dtype)

    scores = einops.einsum(Q_working, K_working, "... query d_k, ... key d_k -> ... query key")
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        mask_bool = mask.to(device=scores.device, dtype=torch.bool)
        if mask_bool.shape != scores.shape:
            mask_bool = torch.broadcast_to(mask_bool, scores.shape)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask_bool, neg_inf)

    weights = softmax(scores, dim=-1).to(working_dtype)
    output = torch.matmul(weights, V_working)
    return output.to(dtype=original_dtype)


def multihead_self_attention(
    num_heads: int,
    Q: Float[torch.Tensor, " ... sequence_length d_model"],
    K: Float[torch.Tensor, " ... sequence_length d_model"],
    V: Float[torch.Tensor, " ... sequence_length d_model"],
    *,
    attn_mask: Bool[torch.Tensor, " ... sequence_length sequence_length"] | None = None,
    rope: RotaryPositionalEmbedding | None = None,
    token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
) -> Float[torch.Tensor, " ... sequence_length d_out"]:
    # if rope is not None:
    #     if token_positions is None:
    #         raise ValueError("token_positions must be provided when using RoPE.")
    #     Q = rope(Q, token_positions)
    #     K = rope(K, token_positions)

    Q_heads = einops.rearrange(Q, "... seq (h dk) -> ... h seq dk", h=num_heads)
    K_heads = einops.rearrange(K, "... seq (h dk) -> ... h seq dk", h=num_heads)
    V_heads = einops.rearrange(V, "... seq (h dv) -> ... h seq dv", h=num_heads)

    if rope is not None:
        if token_positions is None:
            raise ValueError("token_positions must be provided when using RoPE.")
        expanded_positions = einops.repeat(
            token_positions,
            "... seq -> ... h seq",
            h=num_heads,
        )
        Q_heads = rope(Q_heads, expanded_positions)
        K_heads = rope(K_heads, expanded_positions)

    attn_output_heads = scaled_dot_product_attention(Q_heads, K_heads, V_heads, attn_mask)

    attn_output = einops.rearrange(attn_output_heads, "... h seq dv -> ... seq (h dv)")
    return attn_output


class Attn(nn.Module):
    """Multi-head self-attention layer with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        theta: float | None = None,
        max_seq_len: int = 2048,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_k is None:
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads if d_k is not specified.")
            d_k = d_model // num_heads
        if d_v is None:
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads if d_v is not specified.")
            d_v = d_model // num_heads
        if theta is not None and theta <= 0.0:
            raise ValueError("theta must be positive when specified.")

        factory_kwargs: dict[str, object] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.q_proj = Linear(d_model, num_heads * d_k, **factory_kwargs)
        self.k_proj = Linear(d_model, num_heads * d_k, **factory_kwargs)
        self.v_proj = Linear(d_model, num_heads * d_v, **factory_kwargs)
        self.output_proj = Linear(num_heads * d_v, d_model, **factory_kwargs)

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        factory_kwargs["dtype"] = torch.float64
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, **factory_kwargs)
        else:
            self.rope = None

    def forward(
        self,
        x: Float[torch.Tensor, "... seq d_model"],
        token_positions: Int[torch.Tensor, "... seq"] | None = None,
        attn_mask: Bool[torch.Tensor, "... seq seq"] | None = None,
    ) -> Float[torch.Tensor, "... seq d_model"]:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        attn_output = multihead_self_attention(
            self.num_heads,
            Q, K, V,
            attn_mask=attn_mask,
            rope=self.rope,
            token_positions=token_positions,
        )
        output = self.output_proj(attn_output)
        return output


class Block(nn.Module):
    """Transformer block with multi-head self-attention and SwiGLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, object] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.attn = Attn(
            d_model,
            num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            **factory_kwargs,
        )
        self.ffn = SwiGLU(d_model, d_ff=d_ff, **factory_kwargs)

        self.ln1 = RMSNorm(d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, **factory_kwargs)

        self.num_heads = num_heads
        self.d_model = d_model

    def forward(
        self,
        x: Float[torch.Tensor, "... seq d_model"],
        token_positions: Int[torch.Tensor, "... seq"],
        attn_mask: Bool[torch.Tensor, "... seq seq"] | None = None,
    ) -> Float[torch.Tensor, "... seq d_model"]:
        # Multi-head self-attention block
        ## Pre-norm
        x_norm1 = self.ln1(x)

        # print("x after ln1: ", x_norm1)

        attn_output = self.attn(x_norm1, token_positions=token_positions, attn_mask=attn_mask)
 
        # print("x after attn: ", attn_output)

        x = x + attn_output

        # print("x after attn and residual: ", x)

        # Feed-forward network block
        ## Pre-norm
        x_norm2 = self.ln2(x)

        # print("x after ln2: ", x_norm2)

        ffn_output = self.ffn(x_norm2)

        # print("x after ffn: ", ffn_output)

        x = x + ffn_output
        
        # print("x after ffn and residual: ", x)

        return x
        

__all__ = [
    "Linear",
    "Embedding",
    "SwiGLU",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "softmax",
    "scaled_dot_product_attention",
    "multihead_self_attention",
    "Attn",
    "Block",
]