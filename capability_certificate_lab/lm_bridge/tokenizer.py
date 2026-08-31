from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


BYTE_VOCAB_SIZE = 256
PAD_ID = 256
BOS_ID = 257
SEP_ID = 258
EOS_ID = 259
VOCAB_SIZE = 260
MAX_SEQUENCE_LENGTH = 256
MAX_GENERATED_TOKENS = 64

SPECIAL_TOKEN_IDS = frozenset({PAD_ID, BOS_ID, SEP_ID, EOS_ID})


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    prompt: str
    response: str


class ByteTokenizer:
    """Fixed UTF-8 byte tokenizer with four out-of-band special IDs."""

    pad_id = PAD_ID
    bos_id = BOS_ID
    sep_id = SEP_ID
    eos_id = EOS_ID
    vocab_size = VOCAB_SIZE
    max_sequence_length = MAX_SEQUENCE_LENGTH
    max_generated_tokens = MAX_GENERATED_TOKENS

    def encode_text(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a str.")
        return tuple(text.encode("utf-8"))

    def decode_text(self, token_ids: Sequence[int]) -> str:
        for token_id in token_ids:
            self._validate_byte_id(token_id)
        return bytes(token_ids).decode("utf-8", errors="strict")

    def encode_training_record(
        self,
        prompt: str,
        response: str,
        *,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ) -> EncodedRecord:
        self._validate_context_window(max_length)
        prompt_ids = self.encode_text(prompt)
        response_ids = self.encode_text(response)
        input_ids = (BOS_ID, *prompt_ids, SEP_ID, *response_ids, EOS_ID)
        if len(input_ids) > max_length:
            raise ValueError(
                f"Training record length {len(input_ids)} exceeds maximum sequence length {max_length}; "
                "truncation is forbidden."
            )
        self.validate_special_token_placement(input_ids, mode="training")
        return EncodedRecord(input_ids=input_ids, prompt=prompt, response=response)

    def encode_evaluation_prefix(
        self,
        prompt: str,
        *,
        max_length: int = MAX_SEQUENCE_LENGTH,
        max_generated_tokens: int = MAX_GENERATED_TOKENS,
    ) -> tuple[int, ...]:
        self._validate_context_window(max_length)
        if max_generated_tokens < 0:
            raise ValueError("max_generated_tokens must be non-negative.")
        if max_generated_tokens > MAX_GENERATED_TOKENS:
            raise ValueError("Greedy generation is capped at 64 generated tokens.")
        input_ids = (BOS_ID, *self.encode_text(prompt), SEP_ID)
        required = len(input_ids) + max_generated_tokens
        if required > max_length:
            raise ValueError(
                f"Evaluation prefix length violates token budget: prefix {len(input_ids)} "
                f"+ completion {max_generated_tokens} > {max_length}; truncation is forbidden."
            )
        self.validate_special_token_placement(input_ids, mode="evaluation_prefix")
        return input_ids

    def pad(
        self,
        input_ids: Sequence[int],
        length: int,
        *,
        mode: Literal["training", "evaluation_prefix"] = "training",
    ) -> tuple[int, ...]:
        self._validate_context_window(length)
        if len(input_ids) > length:
            raise ValueError(f"Cannot pad sequence of length {len(input_ids)} to shorter length {length}.")
        self.validate_special_token_placement(input_ids, mode=mode)
        return (*tuple(input_ids), *(PAD_ID for _ in range(length - len(input_ids))))

    def batch_pad(
        self,
        records: Sequence[Sequence[int]],
        *,
        length: int | None = None,
        mode: Literal["training", "evaluation_prefix"] = "training",
    ) -> tuple[tuple[int, ...], ...]:
        if not records:
            raise ValueError("records must be non-empty.")
        target_length = length if length is not None else max(len(record) for record in records)
        self._validate_context_window(target_length)
        return tuple(self.pad(record, target_length, mode=mode) for record in records)

    def decode_generated_response(self, full_sequence: Sequence[int]) -> str:
        self._validate_context_window(len(full_sequence))
        self.validate_special_token_placement(full_sequence, mode="generated")
        unpadded = self._strip_padding(full_sequence)
        sep_index = unpadded.index(SEP_ID)
        if EOS_ID in unpadded[sep_index + 1 :]:
            eos_index = unpadded.index(EOS_ID, sep_index + 1)
            response_ids = unpadded[sep_index + 1 : eos_index]
        else:
            response_ids = unpadded[sep_index + 1 :]
        if len(unpadded[sep_index + 1 :]) > MAX_GENERATED_TOKENS:
            raise ValueError("Generated response exceeds the fixed 64-token completion window.")
        return self.decode_text(response_ids)

    def validate_special_token_placement(
        self,
        input_ids: Sequence[int],
        *,
        mode: Literal["training", "evaluation_prefix", "generated"],
    ) -> None:
        self._validate_context_window(len(input_ids))
        if not input_ids:
            raise ValueError("Token sequence must be non-empty.")
        ids = tuple(input_ids)
        for token_id in ids:
            self._validate_token_id(token_id)
        if ids[0] != BOS_ID:
            raise ValueError("BOS must appear exactly at position 0.")

        unpadded = self._strip_padding(ids)
        if PAD_ID in unpadded:
            raise ValueError("PAD may only appear as a suffix.")
        if unpadded.count(BOS_ID) != 1:
            raise ValueError("BOS must appear exactly once.")
        if unpadded.count(SEP_ID) != 1:
            raise ValueError("SEP must appear exactly once.")

        sep_index = unpadded.index(SEP_ID)
        if sep_index == 0:
            raise ValueError("SEP must follow prompt bytes.")
        if mode == "generated" and len(unpadded[sep_index + 1 :]) > MAX_GENERATED_TOKENS:
            raise ValueError("Generated response exceeds the fixed 64-token completion window.")

        eos_count = unpadded.count(EOS_ID)
        if mode == "training":
            if eos_count != 1:
                raise ValueError("Training records must contain exactly one EOS.")
            if unpadded[-1] != EOS_ID:
                raise ValueError("EOS must be the final non-PAD token in training records.")
            if unpadded.index(EOS_ID) <= sep_index:
                raise ValueError("EOS must follow SEP.")
        elif mode == "evaluation_prefix":
            if eos_count:
                raise ValueError("Evaluation prefixes must not contain EOS.")
        elif mode == "generated":
            if eos_count > 1:
                raise ValueError("Generated sequences may contain at most one EOS.")
            if eos_count and unpadded.index(EOS_ID) <= sep_index:
                raise ValueError("Generated EOS must follow SEP.")
            if eos_count and unpadded[-1] != EOS_ID:
                raise ValueError("Generated EOS must be the final non-PAD token.")
        else:
            raise ValueError(f"Unknown validation mode: {mode!r}.")

    def _strip_padding(self, input_ids: Sequence[int]) -> tuple[int, ...]:
        ids = tuple(input_ids)
        end = len(ids)
        while end and ids[end - 1] == PAD_ID:
            end -= 1
        return ids[:end]

    def _validate_byte_id(self, token_id: int) -> None:
        if not isinstance(token_id, int) or not 0 <= token_id < BYTE_VOCAB_SIZE:
            raise ValueError(f"Expected a byte token id in 0..255, got {token_id!r}.")

    def _validate_token_id(self, token_id: int) -> None:
        if not isinstance(token_id, int) or not 0 <= token_id < VOCAB_SIZE:
            raise ValueError(f"Token id must be in 0..{VOCAB_SIZE - 1}, got {token_id!r}.")

    def _validate_context_window(self, length: int) -> None:
        if type(length) is not int:
            raise ValueError("Sequence length limit must be an exact integer.")
        if length < 0:
            raise ValueError("Sequence length limit must be non-negative.")
        if length > MAX_SEQUENCE_LENGTH:
            raise ValueError("Token operations are capped at the fixed 256-token context window.")
