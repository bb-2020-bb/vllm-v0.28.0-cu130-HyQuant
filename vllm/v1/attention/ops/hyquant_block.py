"""Mixed-token HyQuant page packing and decode kernels."""

from __future__ import annotations

import math
import os

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.ops.hyquant_common import (
    HYQUANT_QUANT_FLAG,
    HyQuantBlockLayout,
    get_hyquant_committed_len,
    make_token_map,
    symmetric_int4_groupwise,
    unpack_int4_groupwise,
)
from vllm.v1.attention.ops.hyquant_decode_groupdot import (
    get_hyquant_grouped_decode_status as get_hyquant_groupdot_status,
)
from vllm.v1.attention.ops.hyquant_decode_groupdot import (
    triton_hyquant_groupdot_decode_attention,
)
from vllm.v1.attention.ops.hyquant_decode_grouped import (
    get_hyquant_grouped_decode_status as get_hyquant_legacy_grouped_status,
)
from vllm.v1.attention.ops.hyquant_decode_grouped import (
    triton_hyquant_grouped_decode_attention,
)


def _as_long_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    return values.to(device=device, dtype=torch.long)


@torch.no_grad()
def pack_hyquant_blocks(
    source_key: torch.Tensor,
    source_value: torch.Tensor,
    physical_blocks: torch.Tensor,
    anchor_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    layout: HyQuantBlockLayout,
) -> None:
    """Pack complete logical blocks into compact mixed-token pages.

    ``source_key/value`` are ``[N, B, H, D]`` and ``anchor_indices`` is
    ``[N, A]``. The same token map is used for every KV head, which keeps the
    page metadata small and makes GQA/MQA decode regular.
    """
    if source_key.shape != source_value.shape:
        raise ValueError("HyQuant block K/V source shapes must match")
    if source_key.ndim != 4:
        raise ValueError("HyQuant block sources must be [N, B, H, D]")
    nblocks, block_size, num_heads, head_size = source_key.shape
    if (block_size, head_size) != (layout.block_size, layout.head_size):
        raise ValueError("HyQuant source shape does not match page layout")
    if anchor_indices.shape != (nblocks, layout.anchor_count):
        raise ValueError(
            "HyQuant anchor index shape does not match layout: "
            f"{tuple(anchor_indices.shape)} vs {(nblocks, layout.anchor_count)}"
        )
    if physical_blocks.numel() != nblocks:
        raise ValueError("HyQuant physical block count mismatch")
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 3:
        raise ValueError("HyQuant block cache must be a uint8 page tensor")
    if kv_cache.shape[1] != num_heads or kv_cache.shape[2] != layout.head_page_bytes:
        raise ValueError("HyQuant block cache shape does not match page layout")
    if nblocks == 0:
        return

    device = kv_cache.device
    ids = _as_long_device(physical_blocks, device)
    source_key = source_key.to(device=device, dtype=torch.bfloat16)
    source_value = source_value.to(device=device, dtype=torch.bfloat16)
    anchors = anchor_indices.to(device=device, dtype=torch.long)
    token_map, quant_indices = make_token_map(anchors, block_size)
    source_key = source_key.permute(0, 2, 1, 3).contiguous()
    source_value = source_value.permute(0, 2, 1, 3).contiguous()
    cache_i16 = kv_cache.view(torch.int16)

    if layout.anchor_count:
        anchor_gather = anchors[:, None, :, None].expand(
            nblocks, num_heads, layout.anchor_count, head_size
        )
        key_anchor = torch.gather(source_key, 2, anchor_gather)
        value_anchor = torch.gather(source_value, 2, anchor_gather)
        k_start = layout.anchor_key_offset_bytes // 2
        v_start = layout.anchor_value_offset_bytes // 2
        cache_i16[ids, :, k_start : k_start + layout.anchor_count * head_size] = (
            key_anchor.view(torch.int16).reshape(nblocks, num_heads, -1)
        )
        cache_i16[ids, :, v_start : v_start + layout.anchor_count * head_size] = (
            value_anchor.view(torch.int16).reshape(nblocks, num_heads, -1)
        )

    if layout.quant_count:
        quant_gather = quant_indices[:, None, :, None].expand(
            nblocks, num_heads, layout.quant_count, head_size
        )
        key_quant = torch.gather(source_key, 2, quant_gather)
        value_quant = torch.gather(source_value, 2, quant_gather)
        key_packed, key_scales = symmetric_int4_groupwise(key_quant, layout.group_size)
        value_packed, value_scales = symmetric_int4_groupwise(
            value_quant, layout.group_size
        )
        k_start = layout.quant_key_offset_bytes
        v_start = layout.quant_value_offset_bytes
        kv_cache[
            ids, :, k_start : k_start + layout.quant_count * layout.packed_value_bytes
        ] = key_packed.reshape(nblocks, num_heads, -1)
        kv_cache[
            ids, :, v_start : v_start + layout.quant_count * layout.packed_value_bytes
        ] = value_packed.reshape(nblocks, num_heads, -1)
        k_scale_start = layout.key_scale_offset_bytes // 2
        v_scale_start = layout.value_scale_offset_bytes // 2
        cache_i16[
            ids,
            :,
            k_scale_start : k_scale_start + layout.quant_count * layout.num_groups,
        ] = key_scales.view(torch.int16).reshape(nblocks, num_heads, -1)
        cache_i16[
            ids,
            :,
            v_scale_start : v_scale_start + layout.quant_count * layout.num_groups,
        ] = value_scales.view(torch.int16).reshape(nblocks, num_heads, -1)

    map_start = layout.token_map_offset_bytes // 2
    map_bits = token_map.view(torch.int16)[:, None, :].expand(nblocks, num_heads, -1)
    cache_i16[ids, :, map_start : map_start + block_size] = map_bits


@torch.no_grad()
def materialize_hyquant_blocks(
    kv_cache: torch.Tensor,
    physical_blocks: torch.Tensor,
    layout: HyQuantBlockLayout,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize compact physical pages in logical block order."""
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 3:
        raise ValueError("HyQuant block cache must be a uint8 page tensor")
    if kv_cache.shape[2] != layout.head_page_bytes:
        raise ValueError("HyQuant block cache shape does not match page layout")
    if physical_blocks.ndim != 1:
        raise ValueError("HyQuant physical block IDs must be one-dimensional")

    num_blocks = physical_blocks.numel()
    num_heads = kv_cache.shape[1]
    if num_blocks == 0:
        empty = torch.empty(
            (0, num_heads, layout.head_size),
            dtype=dtype,
            device=kv_cache.device,
        )
        return empty, empty.clone()

    ids = _as_long_device(physical_blocks, kv_cache.device)
    pages = kv_cache.index_select(0, ids)
    pages_i16 = pages.view(torch.int16)
    map_start = layout.token_map_offset_bytes // 2
    codes = (
        pages_i16[:, 0, map_start : map_start + layout.block_size].to(torch.int32)
        & 0xFFFF
    )
    anchor_mask = codes < layout.anchor_count
    anchor_slots = codes.to(torch.long).clamp_max(max(layout.anchor_count - 1, 0))
    quant_slots = (codes.to(torch.long) & (HYQUANT_QUANT_FLAG - 1)).clamp_max(
        max(layout.quant_count - 1, 0)
    )

    if layout.anchor_count:
        anchor_key = (
            pages_i16[
                :,
                :,
                layout.anchor_key_offset_bytes // 2 : layout.anchor_key_offset_bytes
                // 2
                + layout.anchor_count * layout.head_size,
            ]
            .view(torch.bfloat16)
            .view(num_blocks, num_heads, layout.anchor_count, layout.head_size)
        )
        anchor_value = (
            pages_i16[
                :,
                :,
                layout.anchor_value_offset_bytes // 2 : layout.anchor_value_offset_bytes
                // 2
                + layout.anchor_count * layout.head_size,
            ]
            .view(torch.bfloat16)
            .view(num_blocks, num_heads, layout.anchor_count, layout.head_size)
        )
        anchor_gather = anchor_slots[:, None, :, None].expand(
            num_blocks, num_heads, layout.block_size, layout.head_size
        )
        dense_key = torch.gather(anchor_key, 2, anchor_gather).float()
        dense_value = torch.gather(anchor_value, 2, anchor_gather).float()
    else:
        dense_key = torch.zeros(
            num_blocks,
            num_heads,
            layout.block_size,
            layout.head_size,
            device=kv_cache.device,
        )
        dense_value = torch.zeros_like(dense_key)

    if layout.quant_count:
        quant_key = pages[
            :,
            :,
            layout.quant_key_offset_bytes : layout.quant_key_offset_bytes
            + layout.quant_count * layout.packed_value_bytes,
        ].view(
            num_blocks,
            num_heads,
            layout.quant_count,
            layout.packed_value_bytes,
        )
        quant_value = pages[
            :,
            :,
            layout.quant_value_offset_bytes : layout.quant_value_offset_bytes
            + layout.quant_count * layout.packed_value_bytes,
        ].view(
            num_blocks,
            num_heads,
            layout.quant_count,
            layout.packed_value_bytes,
        )
        key_scales = (
            pages_i16[
                :,
                :,
                layout.key_scale_offset_bytes // 2 : layout.key_scale_offset_bytes // 2
                + layout.quant_count * layout.num_groups,
            ]
            .view(torch.float16)
            .view(
                num_blocks,
                num_heads,
                layout.quant_count,
                layout.num_groups,
            )
        )
        value_scales = (
            pages_i16[
                :,
                :,
                layout.value_scale_offset_bytes // 2 : layout.value_scale_offset_bytes
                // 2
                + layout.quant_count * layout.num_groups,
            ]
            .view(torch.float16)
            .view(
                num_blocks,
                num_heads,
                layout.quant_count,
                layout.num_groups,
            )
        )
        packed_gather = quant_slots[:, None, :, None].expand(
            num_blocks,
            num_heads,
            layout.block_size,
            layout.packed_value_bytes,
        )
        scale_gather = quant_slots[:, None, :, None].expand(
            num_blocks,
            num_heads,
            layout.block_size,
            layout.num_groups,
        )
        restored_key = unpack_int4_groupwise(
            torch.gather(quant_key, 2, packed_gather),
            torch.gather(key_scales, 2, scale_gather),
            layout.head_size,
            layout.group_size,
        )
        restored_value = unpack_int4_groupwise(
            torch.gather(quant_value, 2, packed_gather),
            torch.gather(value_scales, 2, scale_gather),
            layout.head_size,
            layout.group_size,
        )
        dense_key = torch.where(anchor_mask[:, None, :, None], dense_key, restored_key)
        dense_value = torch.where(
            anchor_mask[:, None, :, None], dense_value, restored_value
        )

    key = dense_key.permute(0, 2, 1, 3).reshape(
        num_blocks * layout.block_size, num_heads, layout.head_size
    )
    value = dense_value.permute(0, 2, 1, 3).reshape_as(key)
    return key.to(dtype), value.to(dtype)


@torch.no_grad()
def gather_hyquant_block_kv(
    kv_cache: torch.Tensor,
    hot_key: torch.Tensor,
    hot_value: torch.Tensor,
    block_table_row: torch.Tensor,
    sequence_length: int,
    prompt_length: int,
    state_slot: int,
    layout: HyQuantBlockLayout,
    window_size: int,
    retire_interval: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize one logical sequence from compact pages and the hot ring."""
    committed = get_hyquant_committed_len(
        sequence_length,
        prompt_length,
        layout.block_size,
        window_size,
        retire_interval,
    )
    num_heads = kv_cache.shape[1]
    device = kv_cache.device
    key = torch.zeros(
        sequence_length, num_heads, layout.head_size, dtype=torch.float32, device=device
    )
    value = torch.zeros_like(key)
    cache_i16 = kv_cache.view(torch.int16)
    map_start = layout.token_map_offset_bytes // 2
    for logical_block in range(committed // layout.block_size):
        physical = int(block_table_row[logical_block])
        page = kv_cache[physical]
        page_i16 = cache_i16[physical]
        codes = (
            page_i16[0, map_start : map_start + layout.block_size].to(torch.int32)
            & 0xFFFF
        )
        anchor_mask = codes < layout.anchor_count
        anchor_slots = codes.to(torch.long).clamp_max(max(layout.anchor_count - 1, 0))
        quant_slots = (codes.to(torch.long) & (HYQUANT_QUANT_FLAG - 1)).clamp_max(
            max(layout.quant_count - 1, 0)
        )

        if layout.anchor_count:
            anchor_k = (
                page_i16[
                    :,
                    layout.anchor_key_offset_bytes // 2 : layout.anchor_key_offset_bytes
                    // 2
                    + layout.anchor_count * layout.head_size,
                ]
                .view(torch.bfloat16)
                .view(num_heads, layout.anchor_count, layout.head_size)
            )
            anchor_v = (
                page_i16[
                    :,
                    layout.anchor_value_offset_bytes
                    // 2 : layout.anchor_value_offset_bytes // 2
                    + layout.anchor_count * layout.head_size,
                ]
                .view(torch.bfloat16)
                .view(num_heads, layout.anchor_count, layout.head_size)
            )
            dense_k = anchor_k[:, anchor_slots].float()
            dense_v = anchor_v[:, anchor_slots].float()
        else:
            dense_k = torch.zeros(
                num_heads, layout.block_size, layout.head_size, device=device
            )
            dense_v = torch.zeros_like(dense_k)

        if layout.quant_count:
            quant_k = page[
                :,
                layout.quant_key_offset_bytes : layout.quant_key_offset_bytes
                + layout.quant_count * layout.packed_value_bytes,
            ].view(num_heads, layout.quant_count, layout.packed_value_bytes)
            quant_v = page[
                :,
                layout.quant_value_offset_bytes : layout.quant_value_offset_bytes
                + layout.quant_count * layout.packed_value_bytes,
            ].view(num_heads, layout.quant_count, layout.packed_value_bytes)
            scale_k = (
                page_i16[
                    :,
                    layout.key_scale_offset_bytes // 2 : layout.key_scale_offset_bytes
                    // 2
                    + layout.quant_count * layout.num_groups,
                ]
                .view(torch.float16)
                .view(num_heads, layout.quant_count, layout.num_groups)
            )
            scale_v = (
                page_i16[
                    :,
                    layout.value_scale_offset_bytes
                    // 2 : layout.value_scale_offset_bytes // 2
                    + layout.quant_count * layout.num_groups,
                ]
                .view(torch.float16)
                .view(num_heads, layout.quant_count, layout.num_groups)
            )
            restored_k = unpack_int4_groupwise(
                quant_k[:, quant_slots],
                scale_k[:, quant_slots],
                layout.head_size,
                layout.group_size,
            )
            restored_v = unpack_int4_groupwise(
                quant_v[:, quant_slots],
                scale_v[:, quant_slots],
                layout.head_size,
                layout.group_size,
            )
            dense_k = torch.where(anchor_mask[None, :, None], dense_k, restored_k)
            dense_v = torch.where(anchor_mask[None, :, None], dense_v, restored_v)
        positions = slice(
            logical_block * layout.block_size,
            (logical_block + 1) * layout.block_size,
        )
        key[positions] = dense_k.permute(1, 0, 2)
        value[positions] = dense_v.permute(1, 0, 2)

    if sequence_length > committed:
        positions = torch.arange(committed, sequence_length, device=device)
        ring_positions = positions.remainder(hot_key.shape[2])
        key[committed:] = (
            hot_key[state_slot].index_select(1, ring_positions).permute(1, 0, 2).float()
        )
        value[committed:] = (
            hot_value[state_slot]
            .index_select(1, ring_positions)
            .permute(1, 0, 2)
            .float()
        )
    return key, value


@torch.no_grad()
def hyquant_block_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    hot_key: torch.Tensor,
    hot_value: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    state_slots: torch.Tensor,
    layout: HyQuantBlockLayout,
    window_size: int,
    retire_interval: int,
    scale: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference mixed-page decode implementation used as a safe fallback."""
    if output is None:
        output = torch.empty_like(query)
    query_group_size = query.shape[1] // kv_cache.shape[1]
    kv_head_indices = (
        torch.arange(query.shape[1], device=query.device) // query_group_size
    )
    for request_index in range(query.shape[0]):
        key, value = gather_hyquant_block_kv(
            kv_cache,
            hot_key,
            hot_value,
            block_table[request_index],
            int(seq_lens[request_index]),
            int(prompt_lens[request_index]),
            int(state_slots[request_index]),
            layout,
            window_size,
            retire_interval,
        )
        scores = (
            torch.einsum(
                "hd,lhd->hl", query[request_index].float(), key[:, kv_head_indices]
            )
            * scale
        )
        probabilities = torch.softmax(scores, dim=-1)
        output[request_index] = torch.einsum(
            "hl,lhd->hd", probabilities, value[:, kv_head_indices]
        ).to(query.dtype)
    return output


_LAST_BLOCK_DECODE_STATUS = "not_run"
_WARMED_HYQUANT_DECODE_KERNELS: set[tuple] = set()


def get_hyquant_block_decode_status() -> str:
    return _LAST_BLOCK_DECODE_STATUS


if HAS_TRITON:

    @triton.jit
    def _hyquant_mixed_page_decode_kernel(
        query_ptr,
        cache_ptr,
        cache_i16_ptr,
        hot_key_ptr,
        hot_value_ptr,
        block_table_ptr,
        seq_lens_ptr,
        prompt_lens_ptr,
        state_slots_ptr,
        output_ptr,
        stride_query_token,
        stride_query_head,
        stride_cache_block,
        stride_cache_head,
        stride_hot_slot,
        stride_hot_head,
        stride_hot_token,
        stride_block_table,
        stride_output_token,
        stride_output_head,
        num_kv_heads: tl.constexpr,
        query_group_size: tl.constexpr,
        head_size: tl.constexpr,
        block_size: tl.constexpr,
        anchor_count: tl.constexpr,
        window_size: tl.constexpr,
        retire_interval: tl.constexpr,
        hot_capacity: tl.constexpr,
        group_size: tl.constexpr,
        num_groups: tl.constexpr,
        packed_value_bytes: tl.constexpr,
        anchor_key_offset: tl.constexpr,
        anchor_value_offset: tl.constexpr,
        quant_key_offset: tl.constexpr,
        quant_value_offset: tl.constexpr,
        key_scale_offset: tl.constexpr,
        value_scale_offset: tl.constexpr,
        token_map_offset: tl.constexpr,
        block_d: tl.constexpr,
        softmax_scale2: tl.constexpr,
    ):
        request_index = tl.program_id(0)
        query_head = tl.program_id(1)
        kv_head = query_head // query_group_size
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < head_size
        token_offsets = tl.arange(0, block_size)
        query = tl.load(
            query_ptr
            + request_index * stride_query_token
            + query_head * stride_query_head
            + dimensions,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.bfloat16)
        sequence_length = tl.load(seq_lens_ptr + request_index).to(tl.int64)
        prompt_length = tl.load(prompt_lens_ptr + request_index).to(tl.int64)
        initial = tl.maximum(prompt_length - window_size, 0) // block_size * block_size
        generated = tl.maximum(sequence_length - prompt_length, 0)
        completed_intervals = generated // retire_interval
        committed = initial + completed_intervals * retire_interval
        max_allowed = (
            tl.maximum(sequence_length - window_size, 0) // block_size * block_size
        )
        committed = tl.minimum(committed, max_allowed)
        committed_blocks = committed // block_size
        total_blocks = (sequence_length + block_size - 1) // block_size
        block_table_base = request_index * stride_block_table

        max_score = tl.full([], -float("inf"), tl.float32)
        normalizer = tl.zeros([], tl.float32)
        accumulator = tl.zeros([block_d], tl.float32)
        query_row = tl.reshape(query, (1, block_d))

        for block_index in tl.range(0, committed_blocks):
            physical_block = tl.load(
                block_table_ptr + block_table_base + block_index
            ).to(tl.int64)
            page_base = (
                physical_block * stride_cache_block + kv_head * stride_cache_head
            )
            logical_positions = block_index * block_size + token_offsets
            valid = logical_positions < committed
            map_bits = (
                tl.load(
                    cache_i16_ptr + (page_base + token_map_offset) // 2 + token_offsets,
                    mask=valid,
                    other=0x8000,
                ).to(tl.int32)
                & 0xFFFF
            )
            is_anchor = map_bits < anchor_count
            anchor_slots = map_bits
            quant_slots = map_bits & 0x7FFF
            anchor_mask = valid[:, None] & is_anchor[:, None] & dimension_mask[None, :]
            quant_mask = (
                valid[:, None] & (~is_anchor[:, None]) & dimension_mask[None, :]
            )

            anchor_k_bits = tl.load(
                cache_i16_ptr
                + (page_base + anchor_key_offset) // 2
                + anchor_slots[:, None] * head_size
                + dimensions[None, :],
                mask=anchor_mask,
                other=0,
            )
            anchor_v_bits = tl.load(
                cache_i16_ptr
                + (page_base + anchor_value_offset) // 2
                + anchor_slots[:, None] * head_size
                + dimensions[None, :],
                mask=anchor_mask,
                other=0,
            )
            anchor_k = anchor_k_bits.to(tl.bfloat16, bitcast=True)
            anchor_v = anchor_v_bits.to(tl.bfloat16, bitcast=True)

            packed_k = tl.load(
                cache_ptr
                + page_base
                + quant_key_offset
                + quant_slots[:, None] * packed_value_bytes
                + dimensions[None, :] // 2,
                mask=quant_mask,
                other=0,
            ).to(tl.int32)
            nibble_k = tl.where(
                (dimensions[None, :] & 1) == 0,
                packed_k & 0xF,
                (packed_k >> 4) & 0xF,
            )
            scale_k_bits = tl.load(
                cache_i16_ptr
                + (page_base + key_scale_offset) // 2
                + quant_slots[:, None] * num_groups
                + dimensions[None, :] // group_size,
                mask=quant_mask,
                other=0,
            )
            quant_k = (nibble_k << 28 >> 28).to(tl.float32) * scale_k_bits.to(
                tl.float16, bitcast=True
            ).to(tl.float32)
            packed_v = tl.load(
                cache_ptr
                + page_base
                + quant_value_offset
                + quant_slots[:, None] * packed_value_bytes
                + dimensions[None, :] // 2,
                mask=quant_mask,
                other=0,
            ).to(tl.int32)
            nibble_v = tl.where(
                (dimensions[None, :] & 1) == 0,
                packed_v & 0xF,
                (packed_v >> 4) & 0xF,
            )
            scale_v_bits = tl.load(
                cache_i16_ptr
                + (page_base + value_scale_offset) // 2
                + quant_slots[:, None] * num_groups
                + dimensions[None, :] // group_size,
                mask=quant_mask,
                other=0,
            )
            quant_v = (nibble_v << 28 >> 28).to(tl.float32) * scale_v_bits.to(
                tl.float16, bitcast=True
            ).to(tl.float32)
            keys = tl.where(is_anchor[:, None], anchor_k, quant_k).to(tl.bfloat16)
            values = tl.where(is_anchor[:, None], anchor_v, quant_v).to(tl.bfloat16)
            scores = tl.dot(query_row, tl.trans(keys)) * softmax_scale2
            scores = tl.where(valid[None, :], scores, -float("inf"))
            tile_max = tl.max(scores)
            new_max = tl.maximum(max_score, tile_max)
            old_factor = tl.where(
                new_max > -float("inf"), tl.math.exp2(max_score - new_max), 0.0
            )
            probabilities = tl.math.exp2(scores - new_max)
            probabilities = tl.where(valid[None, :], probabilities, 0.0)
            normalizer = normalizer * old_factor + tl.sum(probabilities)
            weighted_values = tl.dot(probabilities.to(tl.bfloat16), values)
            accumulator = accumulator * old_factor + tl.reshape(
                weighted_values, (block_d,)
            ).to(tl.float32)
            max_score = new_max

        state_slot = tl.load(state_slots_ptr + request_index).to(tl.int64)
        for block_index in tl.range(committed_blocks, total_blocks):
            logical_positions = block_index * block_size + token_offsets
            valid = logical_positions < sequence_length
            ring_positions = logical_positions % hot_capacity
            hot_base = (
                state_slot * stride_hot_slot
                + kv_head * stride_hot_head
                + ring_positions[:, None] * stride_hot_token
            )
            keys = tl.load(
                hot_key_ptr + hot_base + dimensions[None, :],
                mask=valid[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            values = tl.load(
                hot_value_ptr + hot_base + dimensions[None, :],
                mask=valid[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            scores = tl.dot(query_row, tl.trans(keys)) * softmax_scale2
            scores = tl.where(valid[None, :], scores, -float("inf"))
            tile_max = tl.max(scores)
            new_max = tl.maximum(max_score, tile_max)
            old_factor = tl.where(
                new_max > -float("inf"), tl.math.exp2(max_score - new_max), 0.0
            )
            probabilities = tl.math.exp2(scores - new_max)
            probabilities = tl.where(valid[None, :], probabilities, 0.0)
            normalizer = normalizer * old_factor + tl.sum(probabilities)
            weighted_values = tl.dot(probabilities.to(tl.bfloat16), values)
            accumulator = accumulator * old_factor + tl.reshape(
                weighted_values, (block_d,)
            ).to(tl.float32)
            max_score = new_max

        tl.store(
            output_ptr
            + request_index * stride_output_token
            + query_head * stride_output_head
            + dimensions,
            accumulator / tl.maximum(normalizer, 1e-20),
            mask=dimension_mask,
        )


def triton_hyquant_block_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    hot_key: torch.Tensor,
    hot_value: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    state_slots: torch.Tensor,
    layout: HyQuantBlockLayout,
    window_size: int,
    retire_interval: int,
    scale: float,
    output: torch.Tensor | None = None,
    mid_o_buf: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    current_key: torch.Tensor | None = None,
    current_value: torch.Tensor | None = None,
    fuse_current_store: bool = False,
) -> torch.Tensor | None:
    """Launch the fused mixed-page decode kernel for standard shapes."""
    global _LAST_BLOCK_DECODE_STATUS
    if not HAS_TRITON or not query.is_cuda:
        _LAST_BLOCK_DECODE_STATUS = "no_triton_or_cuda"
        return None

    # The direct group-dot implementation consumes packed INT4 codes without
    # first materialising a BF16 K/V tile.  It is the preferred production
    # path.  The older grouped materialising kernel remains a compatibility
    # path for layouts that the first group-dot specialization deliberately
    # rejects (for example all-anchor pages).
    if os.environ.get("VLLM_HYQUANT_DISABLE_GROUPDOT", "0") != "1":
        try:
            groupdot_output = triton_hyquant_groupdot_decode_attention(
                query,
                kv_cache,
                hot_key,
                hot_value,
                block_table,
                seq_lens,
                prompt_lens,
                state_slots,
                layout,
                window_size,
                retire_interval,
                scale,
                output=output,
                mid_o_buf=mid_o_buf,
                max_seq_len=max_seq_len,
                current_key=current_key,
                current_value=current_value,
                fuse_current_store=fuse_current_store,
            )
            if groupdot_output is not None:
                _LAST_BLOCK_DECODE_STATUS = (
                    "triton_hyquant_groupdot_splitk"
                    if "splitk" in get_hyquant_groupdot_status()
                    else "triton_hyquant_groupdot"
                )
                return groupdot_output
        except Exception:
            if os.environ.get("VLLM_HYQUANT_GROUPED_STRICT", "0") == "1":
                raise

    # The grouped split-K implementation understands the current compact page
    # ABI and remains a compatibility path while all supported shapes are
    # migrated to group-dot.  It is also useful as an explicit A/B baseline.
    if os.environ.get("VLLM_HYQUANT_DISABLE_GROUPED_DECODE", "0") != "1":
        try:
            grouped_output = triton_hyquant_grouped_decode_attention(
                query,
                kv_cache,
                hot_key,
                hot_value,
                block_table,
                seq_lens,
                prompt_lens,
                state_slots,
                layout,
                window_size,
                retire_interval,
                scale,
                output=output,
                mid_o_buf=mid_o_buf,
                max_seq_len=max_seq_len,
                current_key=current_key,
                current_value=current_value,
                fuse_current_store=fuse_current_store,
            )
            if grouped_output is not None:
                _LAST_BLOCK_DECODE_STATUS = (
                    "triton_hyquant_grouped_splitk"
                    if "splitk" in get_hyquant_legacy_grouped_status()
                    else "triton_hyquant_grouped"
                )
                return grouped_output
        except Exception:
            # Fall through to the proven single-head kernel.  This is also
            # useful during deployment on Triton versions with a missing
            # optional primitive (for example ``tl.interleave``).
            if os.environ.get("VLLM_HYQUANT_GROUPED_STRICT", "0") == "1":
                raise
            if fuse_current_store:
                # The non-grouped compatibility kernel cannot consume the
                # current K/V or publish the fused hot-ring row.  Let the
                # caller perform an explicit store and retry the non-fused
                # path instead of silently returning an incorrect result.
                _LAST_BLOCK_DECODE_STATUS = "grouped_fused_unavailable"
                return None

    if fuse_current_store:
        # A shape/ABI rejection from the grouped launcher has the same
        # requirement as an exception: the legacy kernel is not a valid fused
        # substitute because it expects the current row in the hot ring.
        _LAST_BLOCK_DECODE_STATUS = "grouped_fused_unavailable"
        return None

    if query.ndim != 3 or query.shape[0] != seq_lens.shape[0]:
        _LAST_BLOCK_DECODE_STATUS = "shape"
        return None
    if output is None:
        output = torch.empty_like(query)
    if output.shape != query.shape or output.stride(-1) != 1:
        _LAST_BLOCK_DECODE_STATUS = "output_shape"
        return None
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.shape[2] != layout.head_page_bytes
        or kv_cache.shape[1] <= 0
        or query.shape[1] % kv_cache.shape[1]
    ):
        _LAST_BLOCK_DECODE_STATUS = "cache_shape"
        return None
    if layout.block_size not in (16, 32, 64) or layout.head_size > 256:
        _LAST_BLOCK_DECODE_STATUS = "unsupported_shape"
        return None
    query_group_size = query.shape[1] // kv_cache.shape[1]
    block_d = 1 << (query.shape[-1] - 1).bit_length()
    cache_i16 = kv_cache.view(torch.int16)
    _hyquant_mixed_page_decode_kernel[(query.shape[0], query.shape[1])](
        query,
        kv_cache,
        cache_i16,
        hot_key,
        hot_value,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        output,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        hot_key.stride(0),
        hot_key.stride(1),
        hot_key.stride(2),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        num_kv_heads=kv_cache.shape[1],
        query_group_size=query_group_size,
        head_size=query.shape[-1],
        block_size=layout.block_size,
        anchor_count=layout.anchor_count,
        window_size=window_size,
        retire_interval=retire_interval,
        hot_capacity=hot_key.shape[2],
        group_size=layout.group_size,
        num_groups=layout.num_groups,
        packed_value_bytes=layout.packed_value_bytes,
        anchor_key_offset=layout.anchor_key_offset_bytes,
        anchor_value_offset=layout.anchor_value_offset_bytes,
        quant_key_offset=layout.quant_key_offset_bytes,
        quant_value_offset=layout.quant_value_offset_bytes,
        key_scale_offset=layout.key_scale_offset_bytes,
        value_scale_offset=layout.value_scale_offset_bytes,
        token_map_offset=layout.token_map_offset_bytes,
        block_d=block_d,
        softmax_scale2=scale * 1.4426950408889634,
        num_warps=4 if block_d <= 128 else 8,
        num_stages=2,
    )
    _LAST_BLOCK_DECODE_STATUS = "triton_mixed_token_page"
    return output


def warmup_hyquant_decode_kernel(
    layout: HyQuantBlockLayout,
    num_kv_heads: int,
    num_query_heads: int,
    window_size: int,
    retire_interval: int,
    hot_capacity: int,
    scale: float,
    device: torch.device,
) -> None:
    """Compile the mixed-page decode specialization during model setup."""
    if not HAS_TRITON or device.type != "cuda":
        return
    key = (
        device.index,
        layout,
        num_kv_heads,
        num_query_heads,
        window_size,
        retire_interval,
        hot_capacity,
    )
    if key in _WARMED_HYQUANT_DECODE_KERNELS:
        return
    try:
        cache = torch.zeros(
            1,
            num_kv_heads,
            layout.head_page_bytes,
            dtype=torch.uint8,
            device=device,
        )
        hot_key = torch.zeros(
            1,
            num_kv_heads,
            hot_capacity,
            layout.head_size,
            dtype=torch.bfloat16,
            device=device,
        )
        hot_value = torch.zeros_like(hot_key)
        query = torch.zeros(
            1, num_query_heads, layout.head_size, dtype=torch.bfloat16, device=device
        )
        block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        seq_lens = torch.ones((1,), dtype=torch.int32, device=device)
        prompt_lens = torch.ones_like(seq_lens)
        state_slots = torch.zeros_like(seq_lens)
        result = triton_hyquant_block_decode_attention(
            query,
            cache,
            hot_key,
            hot_value,
            block_table,
            seq_lens,
            prompt_lens,
            state_slots,
            layout,
            window_size,
            retire_interval,
            scale,
        )
        if result is not None:
            _WARMED_HYQUANT_DECODE_KERNELS.add(key)
    except Exception:
        return


def select_important_token_indices(
    source_key: torch.Tensor,
    reference_query: torch.Tensor,
    block_size: int,
    ratio: float,
) -> torch.Tensor:
    """Select important token offsets independently inside every block.

    A two-dimensional reference query is shared across blocks (decode
    retirement). A three-dimensional input supplies one prefix-stable query per
    block (prefill uses the query at that block's final logical token).
    """
    if source_key.ndim != 3 or source_key.shape[0] % block_size:
        raise ValueError(
            "Token selection requires [tokens, kv_heads, head_size] complete blocks"
        )
    num_blocks = source_key.shape[0] // block_size
    count = min(block_size, max(0, math.ceil(block_size * ratio)))
    if num_blocks == 0:
        return torch.empty((0, count), dtype=torch.int16, device=source_key.device)
    if count == 0:
        return torch.empty((num_blocks, 0), dtype=torch.int16, device=source_key.device)
    if count == block_size:
        return (
            torch.arange(block_size, device=source_key.device, dtype=torch.int16)
            .expand(num_blocks, -1)
            .clone()
        )
    num_kv_heads = source_key.shape[1]
    if reference_query.ndim == 2:
        reference_query = reference_query.unsqueeze(0).expand(num_blocks, -1, -1)
    elif reference_query.ndim != 3 or reference_query.shape[0] != num_blocks:
        raise ValueError(
            "Reference queries must be [query_heads, head_size] or "
            "[blocks, query_heads, head_size]"
        )
    if reference_query.shape[1] % num_kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    query_group = reference_query.shape[1] // num_kv_heads
    q = reference_query.reshape(num_blocks, num_kv_heads, query_group, -1).mean(dim=2)
    blocks = source_key.view(num_blocks, block_size, num_kv_heads, -1)
    scores = torch.einsum("nbhd,nhd->nbh", blocks.float(), q.float()).abs().amax(dim=2)
    indices = torch.topk(scores, count, dim=1, largest=True, sorted=False).indices
    return indices.sort(dim=1).values.to(torch.int16)
