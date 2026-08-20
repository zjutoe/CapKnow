from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from capability_certificate_lab.lm_bridge.tokenizer import (
    EOS_ID,
    MAX_GENERATED_TOKENS,
    MAX_SEQUENCE_LENGTH,
    VOCAB_SIZE,
)

TIED_MODEL_PROTOCOL_REVISION = "phase8_tied_io_v1"
LEGACY_UNTIED_MODEL_PROTOCOL_REVISION = "phase8_untied_legacy_v1"


@dataclass(frozen=True)
class TransformerConfig:
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    embedding_weight_tying: bool
    model_protocol_revision: str
    max_seq_len: int = MAX_SEQUENCE_LENGTH
    vocab_size: int = VOCAB_SIZE
    dropout: float = 0.0


SMALL_CONFIG = TransformerConfig(
    name="small",
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    embedding_weight_tying=True,
    model_protocol_revision=TIED_MODEL_PROTOCOL_REVISION,
)
MEDIUM_CONFIG = TransformerConfig(
    name="medium",
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ff=512,
    embedding_weight_tying=True,
    model_protocol_revision=TIED_MODEL_PROTOCOL_REVISION,
)
LEGACY_SMALL_CONFIG = TransformerConfig(
    name="small",
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    embedding_weight_tying=False,
    model_protocol_revision=LEGACY_UNTIED_MODEL_PROTOCOL_REVISION,
)
LEGACY_MEDIUM_CONFIG = TransformerConfig(
    name="medium",
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ff=512,
    embedding_weight_tying=False,
    model_protocol_revision=LEGACY_UNTIED_MODEL_PROTOCOL_REVISION,
)


def transformer_config(name: Literal["small", "medium"]) -> TransformerConfig:
    if name == "small":
        return SMALL_CONFIG
    if name == "medium":
        return MEDIUM_CONFIG
    raise ValueError(f"Unknown frozen model size: {name!r}.")


def historical_transformer_config(name: Literal["small", "medium"]) -> TransformerConfig:
    if name == "small":
        return LEGACY_SMALL_CONFIG
    if name == "medium":
        return LEGACY_MEDIUM_CONFIG
    raise ValueError(f"Unknown historical model size: {name!r}.")


def legacy_serialized_config(name: Literal["small", "medium"]) -> dict[str, object]:
    config = historical_transformer_config(name)
    return {
        "name": config.name,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "d_ff": config.d_ff,
        "max_seq_len": config.max_seq_len,
        "vocab_size": config.vocab_size,
        "dropout": config.dropout,
    }


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device).triu(1)
        normed = self.ln_1(x)
        attended, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + attended
        x = x + self.mlp(self.ln_2(x))
        return x


class ToyCausalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.dropout != 0.0:
            raise ValueError("Phase 8 frozen Transformer dropout must be 0.0.")
        if config.max_seq_len != MAX_SEQUENCE_LENGTH:
            raise ValueError("Phase 8 frozen Transformer context window must be 256 tokens.")
        if config.embedding_weight_tying:
            if config.model_protocol_revision != TIED_MODEL_PROTOCOL_REVISION:
                raise ValueError("Tied Phase 8 models must use protocol revision phase8_tied_io_v1.")
        elif config.model_protocol_revision != LEGACY_UNTIED_MODEL_PROTOCOL_REVISION:
            raise ValueError("Untied Phase 8 models must use the explicit legacy protocol revision.")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(CausalSelfAttentionBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.embedding_weight_tying:
            self.lm_head.weight = self.token_embedding.weight
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds fixed maximum {self.config.max_seq_len}.")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def greedy_decode(
        self,
        prefix_ids: torch.Tensor,
        *,
        max_new_tokens: int = MAX_GENERATED_TOKENS,
        eos_id: int = EOS_ID,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if max_new_tokens > MAX_GENERATED_TOKENS:
            raise ValueError("Greedy decoding is capped at 64 generated tokens.")
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
            raise ValueError("greedy_decode expects a single prefix with shape [1, sequence].")
        if prefix_ids.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("Greedy decoding must fit inside the fixed 256-token context window.")
        was_training = self.training
        self.eval()
        generated = prefix_ids
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_seq_len:
                break
            logits = self(generated)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat((generated, next_id), dim=1)
            if int(next_id.item()) == eos_id:
                break
        if was_training:
            self.train()
        return generated


def build_model(size: Literal["small", "medium"]) -> ToyCausalTransformer:
    return ToyCausalTransformer(transformer_config(size))


def build_historical_model(size: Literal["small", "medium"]) -> ToyCausalTransformer:
    return ToyCausalTransformer(historical_transformer_config(size))
