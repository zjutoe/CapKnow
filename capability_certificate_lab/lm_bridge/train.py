from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
import random
from typing import Literal

import torch
from torch.nn import functional as F

from capability_certificate_lab.lm_bridge.model import (
    ToyCausalTransformer,
    TransformerConfig,
    build_model,
)
from capability_certificate_lab.lm_bridge.tokenizer import (
    BOS_ID,
    EOS_ID,
    MAX_SEQUENCE_LENGTH,
    PAD_ID,
    SEP_ID,
    ByteTokenizer,
)


LEARNING_RATE = 0.0003
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
BATCH_SIZE = 64
TRAINING_STEPS = 1500
MODEL_RNG_OFFSET = 900000
CORPUS_ORDER_RNG_OFFSET = 800000


@dataclass(frozen=True)
class TextRecord:
    prompt: str
    answer: str


@dataclass(frozen=True)
class TrainResult:
    steps: int
    final_loss: float
    training_accuracy: float


def model_seed(seed: int) -> int:
    return MODEL_RNG_OFFSET + seed


def corpus_order_seed(seed: int, state_mask: int) -> int:
    return CORPUS_ORDER_RNG_OFFSET + 100 * seed + state_mask


def set_deterministic_backend(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(model_seed(seed))
    torch.manual_seed(model_seed(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed(seed))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=WEIGHT_DECAY,
    )


def deterministic_batch_indices(
    *,
    record_count: int,
    seed: int,
    state_mask: int,
    batch_size: int = BATCH_SIZE,
    steps: int = TRAINING_STEPS,
) -> tuple[tuple[int, ...], ...]:
    if record_count <= 0:
        raise ValueError("record_count must be positive.")
    rng = random.Random(corpus_order_seed(seed, state_mask))
    order: list[int] = []
    batches: list[tuple[int, ...]] = []
    for _ in range(steps):
        while len(order) < batch_size:
            cycle = list(range(record_count))
            rng.shuffle(cycle)
            order.extend(cycle)
        batches.append(tuple(order[:batch_size]))
        del order[:batch_size]
    return tuple(batches)


def encode_record_batch(
    records: Sequence[TextRecord],
    tokenizer: ByteTokenizer,
    *,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> torch.Tensor:
    encoded = [tokenizer.encode_training_record(record.prompt, record.answer, max_length=max_length).input_ids for record in records]
    padded = tokenizer.batch_pad(encoded, length=max_length, mode="training")
    return torch.tensor(padded, dtype=torch.long)


def response_only_labels(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence].")
    labels = torch.full_like(input_ids, -100)
    tokenizer = ByteTokenizer()
    for row_index, row in enumerate(input_ids.tolist()):
        unpadded_length = len(row)
        while unpadded_length and row[unpadded_length - 1] == PAD_ID:
            unpadded_length -= 1
        unpadded = row[:unpadded_length]
        tokenizer.validate_special_token_placement(unpadded, mode="training")
        if not unpadded or unpadded[0] != BOS_ID:
            raise ValueError("Training sequences must start with BOS.")
        if unpadded.count(SEP_ID) != 1 or unpadded[-1] != EOS_ID:
            raise ValueError("Training sequences must contain one SEP and final EOS before padding.")
        sep_index = unpadded.index(SEP_ID)
        eos_index = len(unpadded) - 1
        if eos_index <= sep_index:
            raise ValueError("EOS must follow SEP.")
        for source_position in range(sep_index, eos_index):
            labels[row_index, source_position] = input_ids[row_index, source_position + 1]
    return labels


def response_only_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    labels = response_only_labels(input_ids).to(logits.device)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def training_accuracy(
    model: ToyCausalTransformer,
    records: Sequence[TextRecord],
    tokenizer: ByteTokenizer,
    *,
    device: torch.device,
) -> float:
    correct = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for record in records:
            prefix = tokenizer.encode_evaluation_prefix(record.prompt)
            prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
            generated = model.greedy_decode(prefix_tensor)
            try:
                decoded = tokenizer.decode_generated_response(tuple(int(token) for token in generated[0].cpu().tolist()))
            except (UnicodeDecodeError, ValueError):
                decoded = None
            if decoded == record.answer:
                correct += 1
    if was_training:
        model.train()
    return correct / len(records)


def train_text_records(
    model: ToyCausalTransformer,
    records: Sequence[TextRecord],
    *,
    seed: int,
    state_mask: int,
    steps: int = TRAINING_STEPS,
    batch_size: int = BATCH_SIZE,
    tokenizer: ByteTokenizer | None = None,
    device: torch.device | None = None,
) -> TrainResult:
    if not records:
        raise ValueError("records must be non-empty.")
    tok = tokenizer or ByteTokenizer()
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.train()
    optimizer = make_optimizer(model)
    batches = deterministic_batch_indices(
        record_count=len(records),
        seed=seed,
        state_mask=state_mask,
        batch_size=batch_size,
        steps=steps,
    )
    final_loss = float("nan")
    for batch in batches:
        batch_records = [records[index] for index in batch]
        input_ids = encode_record_batch(batch_records, tok).to(target_device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = response_only_loss(logits, input_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    accuracy = training_accuracy(model, records, tok, device=target_device)
    return TrainResult(steps=steps, final_loss=final_loss, training_accuracy=accuracy)


def byte_copy_fixture() -> tuple[TextRecord, ...]:
    return (
        TextRecord(prompt="copy a", answer="a"),
        TextRecord(prompt="copy b", answer="b"),
        TextRecord(prompt="copy c", answer="c"),
        TextRecord(prompt="copy d", answer="d"),
    )


def run_byte_copy_overfit_control(
    *,
    seed: int = 0,
    max_steps: int = 500,
    config: TransformerConfig | None = None,
) -> TrainResult:
    set_deterministic_backend(seed)
    if config is None:
        model = build_model("small")
    else:
        model = ToyCausalTransformer(config)
    return train_text_records(
        model,
        byte_copy_fixture(),
        seed=seed,
        state_mask=0,
        steps=max_steps,
        batch_size=4,
        device=torch.device("cpu"),
    )


def save_checkpoint(path: str, model: ToyCausalTransformer, *, metadata: dict[str, object]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "parameter_count": model.parameter_count,
            "metadata": metadata,
        },
        path,
    )


def load_model_from_checkpoint(path: str, *, map_location: str | torch.device = "cpu") -> ToyCausalTransformer:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    config = TransformerConfig(**checkpoint["config"])
    model = ToyCausalTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
