from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable, Callable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor, nn
from multiprocessing import Pool
import regex as re
from collections import Counter
import json
from einops import rearrange, einsum, repeat
import math
from typing import Union, Optional
import numpy as np

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,                   # final dimension of the input
        out_features: int,                  # final dimension of the output
        device: torch.device | None = None, # device to store the parameters on
        dtype: torch.dtype | None = None,   # data type of the parameters
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        sigma = math.sqrt(2.0 / (in_features + out_features))
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(W, 0, sigma, -3.0 * sigma, 3.0 * sigma)
        self.weight = nn.Parameter(W) ## register self.W as a learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx = einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        return Wx


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,       # Size of the vocabulary
        embedding_dim: int,        # Dimension of the embedding vectors, i.e. d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        embed = torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype)
        nn.init.trunc_normal_(embed, 0, 1, -3, 3)
        self.weight = nn.Parameter(embed)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class MyRMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,        # Hidden dimension of the model
        eps: float = 1e-5,   # Epsilon value for numerical stability
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        g = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        ## x should be of shape (bsz, seq, d_model)
        assert x.shape[-1] == self.d_model

        ## calculate rms, with shape (bsz, seq, 1)
        rms = torch.sqrt(torch.mean(torch.pow(x, 2), -1, keepdim=True) + self.eps)
        
        ## calculate output
        output = (x / rms) * self.weight

        return output.to(in_dtype)
    

class MySwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else (d_model // 24) * 64
        self.w1 = Linear(d_model, self.d_ff, device, dtype)
        self.w2 = Linear(self.d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, self.d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        w3x = self.w3(x)
        w1x = w1x * torch.sigmoid(w1x)
        return self.w2(w1x * w3x)
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float, # Θ value for the RoPE
        d_k: int, # dimension of query and key vectors  
        max_seq_len: int, # Maximum sequence length that will be inputted  
        device: torch.device | None = None, # Device to store the buffer on
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        ## register all cos and sin values as a (max_seq_len, d_k // 2, 2) buffer
        ## register_buffer allows the buffer to be moved (across devices) automatically
        ## with the model, but persistent=False means that the buffer is not contained 
        ## in the model's state_dict, so it is not saved to disk, saving memory
        tri = torch.empty(max_seq_len, d_k // 2, 2, device=device)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                theta_ik = 1.0 * i / (theta ** (2.0 * k / d_k))
                cos_theta_ik = math.cos(theta_ik)
                sin_theta_ik = math.sin(theta_ik)
                tri[i, k, 0] = cos_theta_ik
                tri[i, k, 1] = sin_theta_ik
        self.register_buffer("tri", tri, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        ## x is of shape (..., seq_len, d_k)
        ## token_positions is of shape (..., seq_len)
        cos_tri = self.tri[token_positions, :, 0] ## (..., seq_len, d_k // 2)
        sin_tri = self.tri[token_positions, :, 1] ## (..., seq_len, d_k // 2)
        x_shaped = rearrange(x, "... seq (halfdk b) -> ... seq halfdk b", b=2)
        x_rope_0 = x_shaped[..., 0] * cos_tri - x_shaped[..., 1] * sin_tri
        x_rope_1 = x_shaped[..., 0] * sin_tri + x_shaped[..., 1] * cos_tri
        x_rope = rearrange(torch.stack([x_rope_0, x_rope_1], dim=-1), 
                           "... seq halfdk b -> ... seq (halfdk b)")
        return x_rope


class MyMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,            ## Dimensionality of the Transformer block inputs.
        h: int,                  ## Number of heads to use in multi-head self-attention.
        theta: float = 10000,    ## RoPE parameter
        max_seq_len: int = 2048, ## Maximum sequence length
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // h
        self.d_v = d_model // h

        self.q_proj = Linear(d_model, h * self.d_k, device, dtype)
        self.k_proj = Linear(d_model, h * self.d_k, device, dtype)
        self.v_proj = Linear(d_model, h * self.d_v, device, dtype)
        self.output_proj = Linear(h * self.d_v, d_model, device, dtype)

        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)

    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ## x is of shape (bsz, ..., seq, d_model)
        assert x.shape[-1] == self.d_model
        seq = x.shape[-2]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "bsz ... seq (h dk) -> bsz ... h seq dk", h=self.h)
        K = rearrange(K, "bsz ... seq (h dk) -> bsz ... h seq dk", h=self.h)
        V = rearrange(V, "bsz ... seq (h dv) -> bsz ... h seq dv", h=self.h)

        if token_positions is not None: ## apply RoPE
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        mask = torch.tril(torch.ones(seq, seq))
        mask = mask.view((1,) * (Q.ndim - 2) + (seq, seq))  ## 1, ..., 1, seq, seq
        mask = mask.expand(Q.shape[:-2] + (seq, seq))       ## bsz, ..., h, seq, seq
        QKV = run_scaled_dot_product_attention(Q, K, V, mask)
        QKV = rearrange(QKV, "bsz ... h seq dv -> bsz ... seq (h dv)")
        return self.output_proj(QKV)


class MyTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        h: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.ln1 = MyRMSNorm(d_model, eps=0.00001, device=device, dtype=dtype)
        self.attn = MyMultiHeadAttention(d_model, h, theta, max_seq_len, device, dtype)
        self.ln2 = MyRMSNorm(d_model, eps=0.00001, device=device, dtype=dtype)
        self.ffn = MySwiGLU(d_model, d_ff, device, dtype)

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        x = self.ln1(in_features)
        
        bsz, seq = in_features.shape[0], in_features.shape[1]
        token_positions = torch.arange(seq, dtype=torch.int) ## [0, 1, ..., seq - 1]
        token_positions = token_positions.view(1, seq).expand(bsz, seq)
        x = self.attn(x, token_positions)

        in_features += x

        x = self.ln2(in_features)
        x = self.ffn(x)
        in_features += x

        return in_features
    

class MyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        # in_indices: Int[Tensor, " batch_size sequence_length"],
        context_length: int,
        rope_theta: float = 1000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [MyTransformerBlock(d_model, num_heads, d_ff, context_length, \
                                rope_theta, device, dtype) for _ in range(num_layers)]
        )
        self.ln_final = MyRMSNorm(d_model, 0.00001, device, dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) \
        -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        in_features = self.token_embeddings(in_indices) ## (bsz, seq, d_model)
        for i in range(self.num_layers):
            in_features = self.layers[i](in_features)
        in_features = self.ln_final(in_features)
        in_features = self.lm_head(in_features)
        return in_features


class MyAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params, 
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                step = state.get("step", 0)
                beta1_pow = state.get("beta1_pow", 1.0)
                beta2_pow = state.get("beta2_pow", 1.0)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                step += 1
                beta1_pow = beta1_pow * beta1
                beta2_pow = beta2_pow * beta2
                g = p.grad.data
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g ** 2
                lr_biased = lr * math.sqrt(1 - beta2_pow) / (1 - beta1_pow)
                p.data -= lr_biased * m / (torch.sqrt(v) + eps)
                if wd != 0:
                    p.data -= lr * wd * p.data

                state["step"] = step
                state["beta1_pow"] = beta1_pow
                state["beta2_pow"] = beta2_pow
                state["m"] = m
                state["v"] = v

        return loss



def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    linear = Linear(d_in, d_out, weights.device, weights.dtype)
    linear.load_state_dict({"weight": weights}) ## note that state_dict only include nn.param
    out_features = linear(in_features)
    return out_features


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embed = Embedding(vocab_size, d_model, weights.device, weights.dtype)
    embed.load_state_dict({"weight": weights})
    return embed(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    swiglu = MySwiGLU(d_model, d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    swiglu.w1.load_state_dict({"weight": w1_weight})
    swiglu.w2.load_state_dict({"weight": w2_weight})
    swiglu.w3.load_state_dict({"weight": w3_weight})
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "... qnum dk, ... knum dk -> ... qnum knum")
    QK = QK / math.sqrt(d_k)
    QK[mask == False] = -math.inf
    QK = run_softmax(QK, dim=-1)
    QKV = einsum(QK, V, "... qnum knum, ... knum dv -> ... qnum dv")
    return QKV


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    MHA = MyMultiHeadAttention(d_model, num_heads, device=q_proj_weight.device, dtype=q_proj_weight.dtype)
    MHA.q_proj.load_state_dict({"weight": q_proj_weight})
    MHA.k_proj.load_state_dict({"weight": k_proj_weight})
    MHA.v_proj.load_state_dict({"weight": v_proj_weight})
    MHA.output_proj.load_state_dict({"weight": o_proj_weight})
    return MHA(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    MHA = MyMultiHeadAttention(d_model, num_heads, theta, max_seq_len, device=q_proj_weight.device, dtype=q_proj_weight.dtype)
    MHA.q_proj.load_state_dict({"weight": q_proj_weight})
    MHA.k_proj.load_state_dict({"weight": k_proj_weight})
    MHA.v_proj.load_state_dict({"weight": v_proj_weight})
    MHA.output_proj.load_state_dict({"weight": o_proj_weight})
    return MHA(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff). ## Tensor[d_ff, d_model]
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model). ## Tensor[d_model, d_ff]
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff). ## Tensor[d_ff, d_model]
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    device = weights["attn.q_proj.weight"].device
    dtype = weights["attn.q_proj.weight"].dtype
    
    TB = MyTransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device, dtype)
    
    ## the original comments got the shapes of w1, w2 and w3 wrong; so the following is enough
    TB.load_state_dict(weights)
    
    return TB(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff). ## also transposed
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    device = weights["token_embeddings.weight"].device
    dtype = weights["token_embeddings.weight"].dtype
    TF = MyTransformerLM(vocab_size, d_model, num_layers, num_heads, d_ff, context_length, rope_theta, device, dtype)
    ## again, the comments have the wrong shapes of the ffn layer
    TF.load_state_dict(weights)
    return TF(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    
    rmsnorm = MyRMSNorm(d_model, eps, device=weights.device, dtype=weights.dtype)
    rmsnorm.load_state_dict({"weight": weights})
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = dataset.shape[0]
    starts = np.random.randint(0, n - context_length, size=batch_size)[:, None]  ## (b, 1)
    offsets = np.arange(context_length)[None, :]  ## (1, m)
    pos = starts + offsets  ## (b, m)

    ids = dataset[pos]
    tgs = dataset[pos + 1]
    ids = torch.as_tensor(ids, device=device)
    tgs = torch.as_tensor(tgs, device=device)

    print(ids.shape)
    print(tgs.shape)
    return ids, tgs


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_values = in_features.max(dim=dim, keepdim=True)[0] ## contains max and argmax
    in_features = in_features - max_values
    exp_in_features = torch.exp(in_features)
    sum_exp_in_features = exp_in_features.sum(dim=dim, keepdim=True)
    return exp_in_features / sum_exp_in_features


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_values = inputs.max(dim=-1, keepdim=True)[0]
    targets = rearrange(targets, "... seq -> ... seq 1")
    inputs = inputs - max_values
    divisor = inputs.exp().sum(dim=-1)
    target_logits = torch.gather(inputs, -1, targets)
    target_logits = rearrange(target_logits, "... seq 1 -> ... seq") - divisor.log()
    return -target_logits.mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6
    l2n: float = 0
    for param in parameters:
        if param.grad is not None:
            current_l2n = torch.linalg.vector_norm(param.grad).item()
            l2n += current_l2n ** 2
    l2n **= 0.5

    if l2n <= max_l2_norm:
        return
    
    for param in parameters:
        if param.grad is not None:
            param.grad = max_l2_norm / (l2n + eps) * param.grad  


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return MyAdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * \
            (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * \
            (max_learning_rate - min_learning_rate)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()
    obj = {"iteration": iteration}
    obj.update(model_state_dict)
    obj.update(optim_state_dict)
    torch.save(obj, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    obj = torch.load(src)
    model.load_state_dict(obj, strict=False)
    optimizer.load_state_dict(obj)
    return obj["iteration"]


class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        ## note that vocab may not start with ascii
        self.vocab = vocab
        self.merges = merges
        self.vocab_inv: dict[bytes, int] = {tok: tid for tid, tok in vocab.items()}
        self.id_merges: list[tuple[int, int, int]] = []

        for left, right in merges:
            left_id = self.vocab_inv[left]
            right_id = self.vocab_inv[right]
            self.id_merges.append((left_id, right_id, self.vocab_inv[left + right]))    

        if special_tokens is not None:
            ## sanity check: special tokens must be in vocab
            ordered = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens = ordered
            for special_token in ordered:
                if special_token.encode("utf-8") not in self.vocab_inv:
                    raise ValueError(f"## special_token not in vocab: {special_token}")      
        else:
            self.special_tokens = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        """
        Class
        method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens."""
        
        print("############ Constructing Tokenizer from files.")
        print(f"## vocab_filepath: {vocab_filepath}")
        print(f"## merges_filepath: {merges_filepath}")
        
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r") as f: ## vocab.json
            tok_to_id: dict[str, int] = json.load(f)
        vocab = {tid: tok.encode("utf-8") for tok, tid in tok_to_id.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r") as f: ## merges.txt
            for line in f:
                left, right = line.strip().split()
                merges.append((left.encode("utf-8"), right.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        ## pre-tokenize the text

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        if self.special_tokens is not None:
            split_special_token_str = "|".join(re.escape(tok) for tok in self.special_tokens)
            split_special_token_pattern = re.compile(split_special_token_str)
            cnt = Counter()
            for subtext in split_special_token_pattern.split(text):
                # print(f"## subtext: {subtext}")
                for match in re.finditer(PAT, subtext):
                    pretok_str = match.group(0)
                    cnt[pretok_str] += 1
        else:
            cnt = Counter()
            for match in re.finditer(PAT, text):
                pretok_str = match.group(0)
                cnt[pretok_str] += 1

        # print(f"## cnt: {cnt}")

        pretok_to_id = {}
        pretoks = []
        for pretok_str, num in cnt.items():
            pretok_to_id[pretok_str] = len(pretoks)
            pretok_bytes = pretok_str.encode("utf-8")
            pretok = [self.vocab_inv[bytes([b])] for b in pretok_bytes]    
            pretoks.append(PreToken(pretok, num, pretok_str))

        ## -------------------------------------- debug --------------------------------------
        #     if flag:
        #         print(f"## pretok_str: {pretok_str}")
        #         print(f"## pretok: {pretok}")
        #         for b in pretok_bytes:
        #             print(f"## b: {b}, self.vocab_inv[bytes([b])]: {self.vocab_inv[bytes([b])]}")

        # if flag:
        #     print(f"## pretok_to_id: {pretok_to_id}")
        #     print(f"## pretok.tok: {pretoks[0].tok}")
        #     print(f"## cnt: {cnt}")

        # encode the pre-tokens
        for pretok in pretoks:
            for id1, id2, id3 in self.id_merges:
                pretok.only_pop(id1, id2, id3)

        ## merge the encodings
        tokens = []
        if self.special_tokens is not None:
            split_special_token_str_captured = '(' + '|'.join(re.escape(tok) for tok in self.special_tokens) + ')'
            split_special_token_pattern_captured = re.compile(split_special_token_str_captured)
            for sid, subtext in enumerate(split_special_token_pattern_captured.split(text)):
                if sid & 1: ## a special token
                    if len(subtext) > 0: ## eliminate cases where special_token = "" 
                        tokens.append(self.vocab_inv[subtext.encode("utf-8")])
                else:
                    for match in re.finditer(PAT, subtext):
                        pretok_id = pretok_to_id[match.group(0)]
                        tokens += pretoks[pretok_id].tok
                        # if flag:
                        #     print(f"## pretok_id: {pretok_id}")
                        #     print(f"## pretoks[pretok_id].tok: {pretoks[pretok_id].tok}")
        else:
            for match in re.finditer(PAT, text):
                pretok_id = pretok_to_id[match.group(0)]
                tokens += pretoks[pretok_id].tok

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily 
        yields token IDs. This is required for memory-eﬀicient tokenization of large files that we
        cannot directly load into memory.
        """
        for chunk in iterable:
            for tid in self.encode(chunk):
                yield tid
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        ## concatenate self.vocab[id] for id in ids
        bytes_ids = b''.join([self.vocab[id] for id in ids])
        ## decode and handle malformed bytes
        return bytes_ids.decode("utf-8", errors="replace")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    ## -------------------------------------- debug --------------------------------------
    # print(f"## chunk_size: {chunk_size}")  
    # print(f"## file_size: {file_size}")
    # print(f"## desired_num_chunks: {desired_num_chunks}")
    # print(f"## split_special_token: {split_special_token}")

    # ## test finding boundaries bruteforcelly
    # file.seek(0)
    # large_chunk = file.read(file_size)
    # found_at = large_chunk.find(split_special_token)
    # if found_at != -1:
    #     print(f"## found at: {found_at}")
    # else:
    #     print("## not found")
    ## -------------------------------------- debug --------------------------------------

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def tokenize_chunk(chunk: str, special_tokens: list[str]) -> Counter:
    """
    Use special token pattern to split the chunk first, and the use PAT template 
    to pre-tokenize the sub-chunks.
    """
    split_special_token_str = "|".join(re.escape(tok) for tok in special_tokens)
    split_special_token_pattern = re.compile(split_special_token_str)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    cnt = Counter()
    for subchunk in split_special_token_pattern.split(chunk):
        for match in re.finditer(PAT, subchunk):
            tok = match.group(0).encode("utf-8")
            cnt[tok] += 1

    return cnt


class PreToken:
    def __init__(self, pre_token: bytes, num: int, pre_token_str: str | None = None):
        self.num = num                          ## number of occurrences of this pre-token
        self.tok = [b for b in pre_token]       ## tokenization of this pre-token
        self.pre_token_str = pre_token_str      ## reserved for future use

    def pop(self, id1: int, id2: int, id3: int) -> Counter:
        """
        When id1 and id2 are merged into id3, re-compute the tokenization of 
        this pre-token, and return a counter showing the difference.
        """
        diff_counter = Counter()
        new_tok = []
        i = 0
        while i < len(self.tok):
            if i + 1 < len(self.tok) and self.tok[i] == id1 and self.tok[i + 1] == id2:
                new_tok.append(id3)
                if i > 0:
                    diff_counter[(self.tok[i - 1], id3)] += self.num
                    diff_counter[(self.tok[i - 1], id1)] -= self.num
                if i + 2 < len(self.tok):
                    diff_counter[(id3, self.tok[i + 2])] += self.num
                    diff_counter[(id2, self.tok[i + 2])] -= self.num
                i += 2
            else:
                new_tok.append(self.tok[i])
                i += 1
        self.tok = new_tok
        return diff_counter
    
    def only_pop(self, id1: int, id2: int, id3: int):
        """
        pop but does not return counter
        """
        new_tok = []
        i = 0
        while i < len(self.tok):
            if i + 1 < len(self.tok) and self.tok[i] == id1 and self.tok[i + 1] == id2:
                new_tok.append(id3)
                i += 2
            else:
                new_tok.append(self.tok[i])
                i += 1
        self.tok = new_tok
            

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    ## Parallel pre-tokenization, get a (bytes, cnt) counter
    desired_num_processes = min(16, os.cpu_count())
    chunks = []
    assert "<|endoftext|>" in special_tokens, "<|endoftext|> must be contained in special_tokens"
    print(f"## Want to pre-tokenize with {desired_num_processes} processes")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_processes, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    num_processes = len(chunks)
    print(f"## Actually pre-tokenize with {num_processes} processes")
    with Pool(num_processes) as p:
        counters = p.starmap(tokenize_chunk, \
                                    zip(chunks, [special_tokens] * num_processes))
        
    pre_tok_cnt = sum(counters, Counter())
    ## -------------------------------------- debug --------------------------------------
    # print(f"## pre_tok_cnt[:10], {pre_tok_cnt.most_common()[:10]}")

    ## Initialize a pre_token object for all pre-tokens, 
    ## an (int, int) -> int adjacency counter, 
    ## and a dict: (int, int) -> list[int] to record which pre-tokens 
    ## contain that adjacency
    pre_tokens = []
    adj_counter = Counter()
    appear_in : dict[tuple[int, int], list[int]] = dict()
    for tid, (pre_tok_bytes, num) in enumerate(pre_tok_cnt.items()):
        pre_token = PreToken(pre_tok_bytes, num)
        pre_tokens.append(pre_token)
        for i, j in zip(pre_token.tok[:-1], pre_token.tok[1:]):
            adj_counter[(i, j)] += num
            if (i, j) not in appear_in:
                appear_in[(i, j)] = []
            appear_in[(i, j)].append(tid)

    ## -------------------------------------- debug --------------------------------------
    # print("num of pre-tokens: ", tid)
    # for i in range(10):
    #     print(f"## pre_tokens[{i}]: {pre_tokens[i].num}, {pre_tokens[i].tok}")
    # for id, ((i, j), v) in enumerate(adj_counter.most_common()):
    #     print(f"## adj_counter[{i}, {j}]: {v}")
    #     print(f"## appear_in[{i}, {j}]: {appear_in[(i, j)]}")
    #     print(f"## [{i}, {j}] appears in {len(appear_in[(i, j)])} pre-tokens")
    #     if id > 10:
    #         break

    ## BPE merge
    vocab : dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges : list[tuple[bytes, bytes]] = []
    cur_vocab = 256
    for special_token in special_tokens:
        vocab[cur_vocab] = special_token.encode('utf-8')
        cur_vocab += 1
    assert cur_vocab <= vocab_size, "current vocab size must not exceed desired vocab_size"
    print(f"## num_merges: {vocab_size - cur_vocab}")

    while cur_vocab < vocab_size:
        ## find the most frequent adjacency, lexicographically largest
        id1, id2 = max(adj_counter, key=lambda x: (adj_counter[x], vocab[x[0]], vocab[x[1]]))

        if cur_vocab % 100 == 0:
            print(f"## cur_vocab: {cur_vocab} / target: {vocab_size}")
        
        ## -------------------------------------- debug --------------------------------------
        # if cur_vocab == 256 + len(special_tokens) + 92:
        #     print(f"################### cur_vocab: {cur_vocab}")
        #     print(f"## max_adj_tokens: {vocab[id1]}, {vocab[id2]}")
        #     for id, ((i, j), v) in enumerate(adj_counter.most_common()):
        #         print(f"## adj_counter[{vocab[i]}, {vocab[j]}]: {v}")
        #         print(f"## appear_in[{i}, {j}]: {appear_in[(i, j)]}")
        #         print(f"## [{i}, {j}] appears in {len(appear_in[(i, j)])} pre-tokens")
        #         if id > 10:
        #             break
                        
        new_token = vocab[id1] + vocab[id2]
        vocab[cur_vocab] = new_token
        merges.append((vocab[id1], vocab[id2]))
        adj_counter.pop((id1, id2))
        appear_in_pre_tokens = appear_in[(id1, id2)]

        for tid in appear_in_pre_tokens:
            diff_counter = pre_tokens[tid].pop(id1, id2, cur_vocab)
            for (i, j), v in diff_counter.items():
                adj_counter[(i, j)] += v
                if v > 0:   
                    if (i, j) not in appear_in:
                        appear_in[(i, j)] = []
                    appear_in[(i, j)].append(tid)

        cur_vocab += 1

    return vocab, merges
