"""Shared layout and quantization helpers for the HyQuant KV cache.

The page is a fixed-size mixed-token container. A block remains the
PageAttention allocation unit, while importance is decided per token inside
that block. Exactly ``ceil(block_size * top_ratio)`` tokens use BF16 K/V and
the remaining tokens use groupwise symmetric K4/V4. The fixed count makes
the page stride compact and deterministic, without using one scale for a
whole block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

HYQUANT_CACHE_DTYPE = "hyquant_k4v4"
HYQUANT_DEFAULT_TOP_RATIO = 0.0625
HYQUANT_DEFAULT_WINDOW_SIZE = 256
HYQUANT_DEFAULT_RETIRE_INTERVAL = 64
HYQUANT_DEFAULT_GROUP_SIZE = 32
HYQUANT_ALIGNMENT = 16
HYQUANT_QUANT_FLAG = 0x8000
_last_configured_top_ratio = HYQUANT_DEFAULT_TOP_RATIO
_last_configured_group_size = HYQUANT_DEFAULT_GROUP_SIZE


def _align_up(value: int, alignment: int = HYQUANT_ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def get_hyquant_top_ratio() -> float:
    """Read the configured anchor ratio without creating a config cycle."""
    global _last_configured_top_ratio
    try:
        from vllm.config import get_current_vllm_config

        ratio = float(get_current_vllm_config().cache_config.hyquant_top_ratio)
    except (AssertionError, AttributeError, RuntimeError):
        ratio = _last_configured_top_ratio
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"HyQuant top ratio must be in [0, 1], got {ratio}")
    _last_configured_top_ratio = ratio
    return ratio


def get_hyquant_window_config() -> tuple[int, int]:
    """Return ``(BF16 window, decode retirement interval)``."""
    try:
        from vllm.config import get_current_vllm_config

        cache_config = get_current_vllm_config().cache_config
        return (
            int(cache_config.hyquant_window_size),
            int(cache_config.hyquant_retire_interval),
        )
    except (AssertionError, AttributeError, RuntimeError):
        return HYQUANT_DEFAULT_WINDOW_SIZE, HYQUANT_DEFAULT_RETIRE_INTERVAL


def get_hyquant_group_size() -> int:
    """Return the channel group size used inside an INT4 page."""
    global _last_configured_group_size
    try:
        from vllm.config import get_current_vllm_config

        group_size = int(get_current_vllm_config().cache_config.hyquant_group_size)
    except (AssertionError, AttributeError, RuntimeError):
        group_size = _last_configured_group_size
    if group_size <= 0 or group_size % 2:
        raise ValueError(
            f"HyQuant group size must be a positive even number, got {group_size}"
        )
    _last_configured_group_size = group_size
    return group_size


def get_hyquant_hot_capacity(
    window_size: int, retire_interval: int, block_size: int
) -> int:
    """Capacity that prevents overwrite before a periodic retirement."""
    if window_size <= 0 or retire_interval <= 0 or block_size <= 0:
        raise ValueError("HyQuant window, interval, and block size must be positive")
    return window_size + retire_interval + block_size


def get_hyquant_new_completed_block_range(
    context_length: int, sequence_length: int, block_size: int
) -> tuple[int, int]:
    """Return ``[first, end)`` logical blocks completed by a chunk."""
    if context_length < 0 or sequence_length < context_length:
        raise ValueError(
            "Invalid HyQuant chunk bounds: "
            f"context={context_length}, sequence={sequence_length}"
        )
    if block_size <= 0:
        raise ValueError("HyQuant block size must be positive")
    return context_length // block_size, sequence_length // block_size


def get_hyquant_committed_len(
    sequence_length: int,
    prompt_length: int,
    block_size: int,
    window_size: int,
    retire_interval: int,
) -> int:
    """Return the compact-prefix length after this model input token."""
    if block_size <= 0 or window_size <= 0 or retire_interval <= 0:
        raise ValueError("HyQuant block, window, and interval must be positive")
    if sequence_length <= 0:
        return 0
    if prompt_length < 0 or prompt_length > sequence_length:
        raise ValueError(
            f"Invalid prompt/sequence lengths: {prompt_length}/{sequence_length}"
        )
    if window_size % block_size or retire_interval % block_size:
        raise ValueError("HyQuant window and interval must be block aligned")
    initial = max(prompt_length - window_size, 0) // block_size * block_size
    generated = max(sequence_length - prompt_length, 0)
    # Retire the complete interval as one batch.  Splitting the same work
    # across one block per decode step adds a pack/quantization launch at every
    # layer for several consecutive steps and hurts TPOT without reducing the
    # amount of data kept in the hot window.
    completed_intervals = generated // retire_interval
    committed = initial + completed_intervals * retire_interval
    max_allowed = max(sequence_length - window_size, 0) // block_size * block_size
    return min(committed, max_allowed)


@dataclass(frozen=True)
class HyQuantBlockLayout:
    """Fixed mixed-token page layout for one KV head."""

    block_size: int
    head_size: int
    group_size: int
    num_groups: int
    packed_value_bytes: int
    anchor_count: int
    quant_count: int
    anchor_key_offset_bytes: int
    anchor_value_offset_bytes: int
    quant_key_offset_bytes: int
    quant_value_offset_bytes: int
    key_scale_offset_bytes: int
    value_scale_offset_bytes: int
    token_map_offset_bytes: int
    bytes_per_token: int
    head_page_bytes: int

    @classmethod
    def from_params(
        cls,
        block_size: int,
        head_size: int,
        group_size: int | None = None,
        top_ratio: float | None = None,
    ) -> HyQuantBlockLayout:
        if block_size <= 0 or head_size <= 0:
            raise ValueError("block_size and head_size must be positive")
        if head_size % 2:
            raise ValueError("HyQuant K4/V4 requires an even head size")
        if group_size is None:
            group_size = get_hyquant_group_size()
        if group_size <= 0 or group_size % 2:
            raise ValueError("HyQuant group size must be a positive even number")
        if top_ratio is None:
            top_ratio = get_hyquant_top_ratio()
        if not math.isfinite(top_ratio) or not 0.0 <= top_ratio <= 1.0:
            raise ValueError(f"HyQuant top ratio must be in [0, 1], got {top_ratio}")

        num_groups = math.ceil(head_size / group_size)
        packed_value_bytes = math.ceil(head_size / 2)
        anchor_count = min(block_size, max(0, math.ceil(block_size * top_ratio)))
        quant_count = block_size - anchor_count

        anchor_key_offset = 0
        anchor_value_offset = _align_up(
            anchor_key_offset + anchor_count * head_size * 2
        )
        quant_key_offset = _align_up(anchor_value_offset + anchor_count * head_size * 2)
        quant_value_offset = _align_up(
            quant_key_offset + quant_count * packed_value_bytes
        )
        key_scale_offset = _align_up(
            quant_value_offset + quant_count * packed_value_bytes
        )
        value_scale_offset = _align_up(key_scale_offset + quant_count * num_groups * 2)
        token_map_offset = _align_up(value_scale_offset + quant_count * num_groups * 2)
        # ``AttentionSpec.state_content_bytes`` describes one token slot and
        # the allocator derives a page size as ``block_size * slot_bytes``.
        # Align the complete page to the block size so that this derived size
        # is exactly the byte layout exposed by ``get_kv_cache_shape`` even
        # for block sizes larger than the 16-byte metadata alignment.
        head_page_bytes = _align_up(token_map_offset + block_size * 2, block_size)
        return cls(
            block_size=block_size,
            head_size=head_size,
            group_size=group_size,
            num_groups=num_groups,
            packed_value_bytes=packed_value_bytes,
            anchor_count=anchor_count,
            quant_count=quant_count,
            anchor_key_offset_bytes=anchor_key_offset,
            anchor_value_offset_bytes=anchor_value_offset,
            quant_key_offset_bytes=quant_key_offset,
            quant_value_offset_bytes=quant_value_offset,
            key_scale_offset_bytes=key_scale_offset,
            value_scale_offset_bytes=value_scale_offset,
            token_map_offset_bytes=token_map_offset,
            bytes_per_token=math.ceil(head_page_bytes / block_size),
            head_page_bytes=head_page_bytes,
        )

    @property
    def page_bytes_per_head(self) -> int:
        return self.head_page_bytes

    @property
    def page_content_bytes(self) -> int:
        return self.head_page_bytes

    def page_size_bytes(self, num_kv_heads: int) -> int:
        return num_kv_heads * self.head_page_bytes

    @property
    def bf16_bytes_per_token(self) -> int:
        return 4 * self.head_size

    @property
    def quant_bytes_per_token(self) -> int:
        return 2 * self.packed_value_bytes + 4 * self.num_groups

    @property
    def quant_scales_per_block(self) -> int:
        return self.quant_count * self.num_groups

    @property
    def compact_bytes_per_token(self) -> float:
        return self.head_page_bytes / self.block_size


def symmetric_int4_groupwise(
    values: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension with independent symmetric INT4 groups."""
    if values.ndim == 0 or values.shape[-1] % 2:
        raise ValueError("Groupwise INT4 requires an even head size")
    if group_size <= 0 or group_size % 2:
        raise ValueError("Groupwise INT4 requires a positive even group size")
    head_size = values.shape[-1]
    num_groups = math.ceil(head_size / group_size)
    padded = num_groups * group_size
    values_for_scale = (
        torch.nn.functional.pad(values.float(), (0, padded - head_size))
        if padded != head_size
        else values.float()
    )
    grouped = values_for_scale.reshape(*values.shape[:-1], num_groups, group_size)
    scales = grouped.abs().amax(dim=-1).div_(7.0)
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    expanded_scales = scales.repeat_interleave(group_size, -1)[..., :head_size]
    quantized = torch.round(values.float() / expanded_scales).clamp_(-8, 7)
    return pack_int4(quantized), scales.to(torch.float16)


def unpack_int4_groupwise(
    packed: torch.Tensor, scales: torch.Tensor, head_size: int, group_size: int
) -> torch.Tensor:
    if head_size <= 0 or head_size % 2 or group_size <= 0 or group_size % 2:
        raise ValueError("Groupwise INT4 requires positive even dimensions")
    values = unpack_int4(packed, head_size)
    num_groups = math.ceil(head_size / group_size)
    if scales.shape[-1] != num_groups:
        raise ValueError("Invalid groupwise scale shape")
    expanded = scales.float().repeat_interleave(group_size, dim=-1)[..., :head_size]
    return values * expanded


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed symmetric int4 values, low/even nibble first."""
    signed = values.to(torch.int16).clamp_(-8, 7)
    if signed.shape[-1] % 2:
        signed = torch.nn.functional.pad(signed, (0, 1))
    low = signed[..., 0::2] & 0xF
    high = (signed[..., 1::2] & 0xF) << 4
    return (low | high).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, head_size: int) -> torch.Tensor:
    """Unpack signed int4 bytes to float32 values."""
    packed_i16 = packed.to(torch.int16)
    low = packed_i16 & 0xF
    high = (packed_i16 >> 4) & 0xF
    values = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    return torch.where(values >= 8, values - 16, values)[..., :head_size].float()


def make_token_map(
    anchor_indices: torch.Tensor,
    block_size: int,
    quant_flag: int = HYQUANT_QUANT_FLAG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a uint16 token map and complementary quantized indices."""
    if anchor_indices.ndim != 2 or anchor_indices.shape[1] > block_size:
        raise ValueError("Invalid HyQuant anchor index shape")
    nblocks, anchor_count = anchor_indices.shape
    device = anchor_indices.device
    signed_quant_flag = quant_flag - 0x10000
    if anchor_count == 0:
        quant_indices = torch.arange(block_size, device=device, dtype=torch.long)
        quant_indices = quant_indices.expand(nblocks, -1)
        token_map = torch.full(
            (nblocks, block_size), signed_quant_flag, device=device, dtype=torch.int16
        ).view(torch.uint16)
        quant_codes = (
            torch.arange(block_size, device=device, dtype=torch.int16)
            .add(signed_quant_flag)
            .expand(nblocks, -1)
        )
        token_map = quant_codes.contiguous().view(torch.uint16)
        return token_map, quant_indices
    anchors = anchor_indices.to(device=device, dtype=torch.long)
    if bool((anchors < 0).any()) or bool((anchors >= block_size).any()):
        raise ValueError("HyQuant anchor index out of range")
    anchor_mask = torch.zeros(nblocks, block_size, dtype=torch.bool, device=device)
    anchor_mask.scatter_(1, anchors, True)
    if bool(anchor_mask.sum(dim=1).ne(anchor_count).any()):
        raise ValueError("HyQuant anchor indices must be unique per block")
    token_map = torch.full(
        (nblocks, block_size), signed_quant_flag, device=device, dtype=torch.int16
    )
    token_map.scatter_(
        1,
        anchors,
        torch.arange(anchor_count, device=device, dtype=torch.int16).expand_as(anchors),
    )
    quant_indices = (
        torch.arange(block_size, device=device)
        .expand(nblocks, -1)[~anchor_mask]
        .view(nblocks, block_size - anchor_count)
    )
    quant_codes = (
        torch.arange(block_size - anchor_count, device=device, dtype=torch.int16)
        .add(signed_quant_flag)
        .expand(nblocks, -1)
    )
    token_map.scatter_(1, quant_indices, quant_codes)
    return token_map.view(torch.uint16), quant_indices
