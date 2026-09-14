"""Grouped split-K decode kernels for the compact HyQuant page format.

The regular HyQuant kernel assigns one program to every query head and scans
the complete committed prefix serially.  That is a poor match for GQA decode:
the same KV page is reread by every query head and a small batch does not
provide enough independent programs.  This module keeps the page ABI intact,
groups query heads belonging to one KV head, and partitions the committed
prefix into independent softmax reductions.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.ops.hyquant_common import HyQuantBlockLayout

_LOG2_E = 1.4426950408889634
_LAST_STATUS = "not_run"


def get_hyquant_grouped_decode_status() -> str:
    return _LAST_STATUS


def get_hyquant_decode_split_count(
    batch_size: int, max_seq_len: int, query_group_size: int
) -> int:
    """Choose a deterministic split count for the packed decode scan.

    The environment override is useful for per-GPU tuning and does not alter
    the page format.  MHA deliberately stays unsplit because it already has
    one independent program per query head.
    """
    override = os.environ.get("VLLM_HYQUANT_DECODE_SPLITS")
    if override:
        try:
            value = int(override)
        except ValueError as exc:
            raise ValueError(
                "VLLM_HYQUANT_DECODE_SPLITS must be an integer"
            ) from exc
        if value not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError(
                "VLLM_HYQUANT_DECODE_SPLITS must be one of 1,2,4,8,16,32,64"
            )
        return value
    if batch_size <= 0 or query_group_size <= 1 or max_seq_len < 512:
        return 1
    if batch_size <= 2:
        # A single long request otherwise leaves only eight active KV-head
        # programs at each split.  32-way split-K supplies enough resident
        # programs on SM80; the extra page reads are cheaper than serial tails.
        return 32 if max_seq_len >= 8192 else (16 if max_seq_len >= 2048 else 4)
    if batch_size <= 8:
        return 16 if max_seq_len >= 4096 else (8 if max_seq_len >= 2048 else 4)
    if batch_size <= 16:
        return 16 if max_seq_len >= 6144 else 8
    return 16 if max_seq_len >= 4096 else 8


def get_hyquant_decode_launch_config(
    batch_size: int, max_seq_len: int, query_group_size: int
) -> tuple[int, int, int]:
    """Return ``(BLOCK_KV, BLOCK_H, num_warps)`` for the packed kernel."""
    # A 128-token tile amortizes the packed-page address/decode work on SM80.
    # At larger batches the extra registers reduce occupancy, so retain the
    # smaller tile there.
    block_kv = (
        128 if batch_size <= 8 and max_seq_len >= 2048
        else 64 if batch_size <= 8
        else 32
    )
    if query_group_size <= 1:
        block_h = 1
    elif query_group_size <= 4:
        block_h = query_group_size
    elif query_group_size <= 8:
        block_h = 4
    else:
        block_h = 8
    # The 4-warp schedule is faster for the long-context GQA shape on A800 and
    # avoids the register pressure of the old 8-warp default.  Keep the value
    # explicit so deployments can still override it for another GPU.
    num_warps = 4
    block_kv_override = os.environ.get("VLLM_HYQUANT_DECODE_BLOCK_KV")
    if block_kv_override:
        block_kv = int(block_kv_override)
        if block_kv not in (32, 64, 128, 256):
            raise ValueError("VLLM_HYQUANT_DECODE_BLOCK_KV must be 32/64/128/256")
    block_h_override = os.environ.get("VLLM_HYQUANT_DECODE_BLOCK_H")
    if block_h_override:
        block_h = int(block_h_override)
        if block_h not in (1, 2, 4, 8, 16):
            raise ValueError("VLLM_HYQUANT_DECODE_BLOCK_H must be 1/2/4/8/16")
    warp_override = os.environ.get("VLLM_HYQUANT_DECODE_WARPS")
    if warp_override:
        num_warps = int(warp_override)
        if num_warps not in (1, 2, 4, 8):
            raise ValueError("VLLM_HYQUANT_DECODE_WARPS must be 1/2/4/8")
    return block_kv, block_h, num_warps


if HAS_TRITON:

    @triton.jit
    def _hyquant_grouped_split_decode_kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        cache_ptr,
        cache_i16_ptr,
        hot_key_ptr,
        hot_value_ptr,
        block_table_ptr,
        seq_lens_ptr,
        prompt_lens_ptr,
        state_slots_ptr,
        output_ptr,
        mid_o_ptr,
        stride_query_token,
        stride_query_head,
        stride_current_key_token,
        stride_current_key_head,
        stride_current_value_token,
        stride_current_value_head,
        stride_cache_block,
        stride_cache_head,
        stride_hot_slot,
        stride_hot_head,
        stride_hot_token,
        stride_block_table,
        stride_output_token,
        stride_output_head,
        stride_mid_batch,
        stride_mid_head,
        stride_mid_split,
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
        block_h: tl.constexpr,
        block_half: tl.constexpr,
        block_packed_half: tl.constexpr,
        block_g: tl.constexpr,
        block_d: tl.constexpr,
        block_kv: tl.constexpr,
        block_table_tile: tl.constexpr,
        num_splits: tl.constexpr,
        coalesce_block_table: tl.constexpr,
        fuse_current_store: tl.constexpr,
        softmax_scale2: tl.constexpr,
    ):
        request_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        group_split = tl.program_id(2)
        split_index = group_split % num_splits
        group_index = group_split // num_splits

        group_offsets = tl.arange(0, block_h)
        group_start = group_index * block_h
        query_heads = kv_head * query_group_size + group_start + group_offsets
        group_mask = group_offsets < (query_group_size - group_start)

        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < head_size
        half_offsets = tl.arange(0, block_half)
        half_mask = half_offsets < packed_value_bytes
        group_ids = tl.minimum(dimensions // group_size, num_groups - 1)
        query_bases = (
            request_index * stride_query_token
            + query_heads[:, None] * stride_query_head
        )
        query_rows = tl.load(
            query_ptr + query_bases + dimensions[None, :],
            mask=group_mask[:, None] & dimension_mask[None, :],
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)

        sequence_length = tl.load(seq_lens_ptr + request_index).to(tl.int64)
        prompt_length = tl.load(prompt_lens_ptr + request_index).to(tl.int64)
        state_slot = tl.load(state_slots_ptr + request_index).to(tl.int64)
        initial = tl.maximum(prompt_length - window_size, 0) // block_size * block_size
        generated = tl.maximum(sequence_length - prompt_length, 0)
        completed_intervals = generated // retire_interval
        committed = initial + completed_intervals * retire_interval
        max_allowed = (
            tl.maximum(sequence_length - window_size, 0) // block_size * block_size
        )
        committed = tl.minimum(committed, max_allowed)
        committed_blocks = committed // block_size
        block_table_base = request_index * stride_block_table

        max_score = tl.full([block_h], -float("inf"), tl.float32)
        normalizer = tl.zeros([block_h], tl.float32)
        accumulator = tl.zeros([block_h, block_d], tl.float32)
        kv_offsets = tl.arange(0, block_kv)

        # Quantized rows are contiguous in each compact page.  This avoids the
        # token-map lookup and lets a split consume a contiguous range of rows.
        if anchor_count < block_size:
            quant_per_block = block_size - anchor_count
            quant_count = committed_blocks * quant_per_block
            quant_tiles = tl.cdiv(quant_count, block_kv)
            tiles_per_split = tl.cdiv(quant_tiles, num_splits)
            quant_start = split_index * tiles_per_split * block_kv
            quant_end = tl.minimum(
                quant_start + tiles_per_split * block_kv, quant_count
            )
            for start in tl.range(quant_start, quant_end, block_kv):
                quant_indices = start + kv_offsets
                valid = quant_indices < quant_count
                logical_blocks = quant_indices // quant_per_block
                quant_rows = quant_indices % quant_per_block
                if coalesce_block_table:
                    # A block contributes ``quant_per_block`` rows.  Loading
                    # the page table once per distinct block avoids issuing
                    # the same pointer load for every lane in a KV tile.
                    first_block = start // quant_per_block
                    table_offsets = tl.arange(0, block_table_tile)
                    table_valid = (
                        (table_offsets < tl.cdiv(block_kv, quant_per_block))
                        & (first_block + table_offsets < committed_blocks)
                    )
                    table_values = tl.load(
                        block_table_ptr
                        + block_table_base
                        + first_block
                        + table_offsets,
                        mask=table_valid,
                        other=0,
                    ).to(tl.int64)
                    safe_select = tl.minimum(
                        logical_blocks - first_block, block_table_tile - 1
                    )
                    physical_blocks = tl.gather(
                        table_values, safe_select, axis=0
                    )
                else:
                    safe_blocks = tl.where(valid, logical_blocks, 0)
                    physical_blocks = tl.load(
                        block_table_ptr + block_table_base + safe_blocks,
                        mask=valid,
                        other=0,
                    ).to(tl.int64)
                page_bases = (
                    physical_blocks * stride_cache_block + kv_head * stride_cache_head
                )

                if packed_value_bytes % 2 == 0:
                    packed_key_words = tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + quant_key_offset) // 2
                        + quant_rows[:, None] * (packed_value_bytes // 2)
                        + tl.arange(0, block_packed_half)[None, :],
                        mask=valid[:, None]
                        & (tl.arange(0, block_packed_half)[None, :]
                           < (packed_value_bytes // 2)),
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32) & 0xFFFF
                    key_n0 = ((packed_key_words & 0xF) ^ 0x8) - 0x8
                    key_n1 = (((packed_key_words >> 4) & 0xF) ^ 0x8) - 0x8
                    key_n2 = (((packed_key_words >> 8) & 0xF) ^ 0x8) - 0x8
                    key_n3 = (((packed_key_words >> 12) & 0xF) ^ 0x8) - 0x8
                    key_codes = tl.reshape(
                        tl.join(
                            tl.join(key_n0, key_n2),
                            tl.join(key_n1, key_n3),
                        ),
                        (block_kv, block_d),
                    )
                else:
                    packed_key = tl.load(
                        cache_ptr
                        + page_bases[:, None]
                        + quant_key_offset
                        + quant_rows[:, None] * packed_value_bytes
                        + half_offsets[None, :],
                        mask=valid[:, None] & half_mask[None, :],
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32)
                    key_low = ((packed_key & 0xF) ^ 0x8) - 0x8
                    key_high = (((packed_key >> 4) & 0xF) ^ 0x8) - 0x8
                    key_codes = tl.interleave(key_low, key_high)
                if num_groups * group_size == block_d:
                    scale_offsets = tl.arange(0, block_g)
                    scale_mask = scale_offsets < num_groups
                    key_scale_rows = tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + key_scale_offset) // 2
                        + quant_rows[:, None] * num_groups
                        + scale_offsets[None, :],
                        mask=valid[:, None] & scale_mask[None, :],
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.float16, bitcast=True).to(tl.bfloat16)
                    key_grouped = tl.reshape(
                        key_codes.to(tl.bfloat16), (block_kv, num_groups, group_size)
                    )
                    keys = tl.reshape(
                        key_grouped * key_scale_rows[:, :, None],
                        (block_kv, block_d),
                    ).to(tl.bfloat16)
                else:
                    key_scale_bits = tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + key_scale_offset) // 2
                        + quant_rows[:, None] * num_groups
                        + group_ids[None, :],
                        mask=valid[:, None] & dimension_mask[None, :],
                        other=0,
                        cache_modifier=".cg",
                    )
                    key_scales = key_scale_bits.to(
                        tl.float16, bitcast=True
                    ).to(tl.float32)
                    keys = (key_codes.to(tl.float32) * key_scales).to(tl.bfloat16)
                scores = tl.dot(query_rows, tl.trans(keys)) * softmax_scale2
                score_mask = group_mask[:, None] & valid[None, :]
                scores = tl.where(score_mask, scores, -float("inf"))
                tile_max = tl.max(scores, axis=1)
                new_max = tl.maximum(max_score, tile_max)
                old_factor = tl.where(
                    new_max > -float("inf"),
                    tl.math.exp2(max_score - new_max),
                    0.0,
                )
                probabilities = tl.math.exp2(scores - new_max[:, None])
                probabilities = tl.where(score_mask, probabilities, 0.0)
                normalizer = normalizer * old_factor + tl.sum(probabilities, axis=1)

                if packed_value_bytes % 2 == 0:
                    packed_value_words = tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + quant_value_offset) // 2
                        + quant_rows[:, None] * (packed_value_bytes // 2)
                        + tl.arange(0, block_packed_half)[None, :],
                        mask=valid[:, None]
                        & (tl.arange(0, block_packed_half)[None, :]
                           < (packed_value_bytes // 2)),
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32) & 0xFFFF
                    value_n0 = ((packed_value_words & 0xF) ^ 0x8) - 0x8
                    value_n1 = (((packed_value_words >> 4) & 0xF) ^ 0x8) - 0x8
                    value_n2 = (((packed_value_words >> 8) & 0xF) ^ 0x8) - 0x8
                    value_n3 = (((packed_value_words >> 12) & 0xF) ^ 0x8) - 0x8
                    value_codes = tl.reshape(
                        tl.join(
                            tl.join(value_n0, value_n2),
                            tl.join(value_n1, value_n3),
                        ),
                        (block_kv, block_d),
                    )
                else:
                    packed_value = tl.load(
                        cache_ptr
                        + page_bases[:, None]
                        + quant_value_offset
                        + quant_rows[:, None] * packed_value_bytes
                        + half_offsets[None, :],
                        mask=valid[:, None] & half_mask[None, :],
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32)
                    value_low = ((packed_value & 0xF) ^ 0x8) - 0x8
                    value_high = (((packed_value >> 4) & 0xF) ^ 0x8) - 0x8
                    value_codes = tl.interleave(value_low, value_high)
                if num_groups * group_size == block_d:
                    scale_offsets = tl.arange(0, block_g)
                    scale_mask = scale_offsets < num_groups
                    value_scale_rows = tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + value_scale_offset) // 2
                        + quant_rows[:, None] * num_groups
                        + scale_offsets[None, :],
                        mask=valid[:, None] & scale_mask[None, :],
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.float16, bitcast=True).to(tl.bfloat16)
                    value_grouped = tl.reshape(
                        value_codes.to(tl.bfloat16), (block_kv, num_groups, group_size)
                    )
                    values = tl.reshape(
                        value_grouped * value_scale_rows[:, :, None],
                        (block_kv, block_d),
                    ).to(tl.bfloat16)
                else:
                    value_scale_bits = tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + value_scale_offset) // 2
                        + quant_rows[:, None] * num_groups
                        + group_ids[None, :],
                        mask=valid[:, None] & dimension_mask[None, :],
                        other=0,
                    )
                    value_scales = value_scale_bits.to(
                        tl.float16, bitcast=True
                    ).to(tl.float32)
                    values = (value_codes.to(tl.float32) * value_scales).to(tl.bfloat16)
                accumulator = accumulator * old_factor[:, None] + tl.dot(
                    probabilities.to(tl.bfloat16), values
                )
                max_score = new_max

        # Anchors are contiguous BF16 rows in the page and share the same
        # online-softmax state as the quantized rows.
        if anchor_count > 0:
            anchor_total = committed_blocks * anchor_count
            anchor_tiles = tl.cdiv(anchor_total, block_kv)
            tiles_per_split = tl.cdiv(anchor_tiles, num_splits)
            anchor_start = split_index * tiles_per_split * block_kv
            anchor_end = tl.minimum(
                anchor_start + tiles_per_split * block_kv, anchor_total
            )
            for start in tl.range(anchor_start, anchor_end, block_kv):
                anchor_indices = start + kv_offsets
                valid = anchor_indices < anchor_total
                logical_blocks = anchor_indices // anchor_count
                anchor_rows = anchor_indices % anchor_count
                safe_blocks = tl.where(valid, logical_blocks, 0)
                physical_blocks = tl.load(
                    block_table_ptr + block_table_base + safe_blocks,
                    mask=valid,
                    other=0,
                ).to(tl.int64)
                page_bases = (
                    physical_blocks * stride_cache_block + kv_head * stride_cache_head
                )
                key_bases = (
                    (page_bases + anchor_key_offset) // 2
                    + anchor_rows * head_size
                )
                keys = tl.load(
                    cache_i16_ptr
                    + key_bases[:, None]
                    + dimensions[None, :],
                    mask=valid[:, None] & dimension_mask[None, :],
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.bfloat16, bitcast=True)
                scores = tl.dot(query_rows, tl.trans(keys)) * softmax_scale2
                score_mask = group_mask[:, None] & valid[None, :]
                scores = tl.where(score_mask, scores, -float("inf"))
                tile_max = tl.max(scores, axis=1)
                new_max = tl.maximum(max_score, tile_max)
                old_factor = tl.where(
                    new_max > -float("inf"),
                    tl.math.exp2(max_score - new_max),
                    0.0,
                )
                probabilities = tl.math.exp2(scores - new_max[:, None])
                probabilities = tl.where(score_mask, probabilities, 0.0)
                normalizer = normalizer * old_factor + tl.sum(probabilities, axis=1)
                value_bases = (
                    (page_bases + anchor_value_offset) // 2
                    + anchor_rows * head_size
                )
                values = tl.load(
                    cache_i16_ptr
                    + value_bases[:, None]
                    + dimensions[None, :],
                    mask=valid[:, None] & dimension_mask[None, :],
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.bfloat16, bitcast=True)
                accumulator = accumulator * old_factor[:, None] + tl.dot(
                    probabilities.to(tl.bfloat16), values
                )
                max_score = new_max

        # The current decode row is normally written by a separate kernel
        # before attention.  In the fused mode, scan only the already-published
        # ring rows here and consume the current K/V once in its owning split.
        # This removes one launch and avoids writing then immediately rereading
        # the current row from the hot ring.
        hot_count = sequence_length - committed
        hot_scan_count = (
            tl.maximum(hot_count - 1, 0) if fuse_current_store else hot_count
        )
        hot_tiles = tl.cdiv(hot_scan_count, block_kv)
        tiles_per_split = tl.cdiv(hot_tiles, num_splits)
        hot_start = split_index * tiles_per_split * block_kv
        hot_end = tl.minimum(
            hot_start + tiles_per_split * block_kv, hot_scan_count
        )
        if fuse_current_store:
            current_owner = tl.minimum(
                tl.maximum(hot_count - 1, 0)
                // tl.maximum(tiles_per_split * block_kv, 1),
                num_splits - 1,
            )
        for start in tl.range(hot_start, hot_end, block_kv):
            hot_indices = start + kv_offsets
            valid = hot_indices < hot_scan_count
            logical_positions = committed + hot_indices
            ring_positions = logical_positions % hot_capacity
            hot_bases = (
                state_slot * stride_hot_slot
                + kv_head * stride_hot_head
                + ring_positions[:, None] * stride_hot_token
            )
            keys = tl.load(
                hot_key_ptr + hot_bases + dimensions[None, :],
                mask=valid[:, None] & dimension_mask[None, :],
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.bfloat16)
            scores = tl.dot(query_rows, tl.trans(keys)) * softmax_scale2
            score_mask = group_mask[:, None] & valid[None, :]
            scores = tl.where(score_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_score, tile_max)
            old_factor = tl.where(
                new_max > -float("inf"),
                tl.math.exp2(max_score - new_max),
                0.0,
            )
            probabilities = tl.math.exp2(scores - new_max[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            normalizer = normalizer * old_factor + tl.sum(probabilities, axis=1)
            values = tl.load(
                hot_value_ptr + hot_bases + dimensions[None, :],
                mask=valid[:, None] & dimension_mask[None, :],
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.bfloat16)
            accumulator = accumulator * old_factor[:, None] + tl.dot(
                probabilities.to(tl.bfloat16), values
            )
            max_score = new_max

        if fuse_current_store:
            # Every query-group tile needs the current row for its own softmax,
            # but only one split contributes it to the split-K reduction.  The
            # group-zero tile is also the sole writer for the shared hot ring.
            # Keep the owner predicate on the loads themselves.  Previously
            # every split fetched the same current K/V row and discarded it
            # after the score mask, which is especially wasteful at 32-way
            # split-K for a single long request.
            current_key_base = (
                request_index * stride_current_key_token
                + kv_head * stride_current_key_head
            )
            current_value_base = (
                request_index * stride_current_value_token
                + kv_head * stride_current_value_head
            )
            current_active = (state_slot >= 0) & (split_index == current_owner)
            current_key_row = tl.load(
                current_key_ptr + current_key_base + dimensions,
                mask=current_active & dimension_mask,
                other=0.0,
            ).to(tl.bfloat16)
            current_scores = tl.sum(
                query_rows.to(tl.float32)
                * current_key_row[None, :].to(tl.float32),
                axis=1,
            ) * softmax_scale2
            current_mask = (
                group_mask
                & (state_slot >= 0)
                & (split_index == current_owner)
            )
            current_scores = tl.where(
                current_mask, current_scores, -float("inf")
            )
            new_max = tl.maximum(max_score, current_scores)
            old_factor = tl.where(
                new_max > -float("inf"),
                tl.math.exp2(max_score - new_max),
                0.0,
            )
            current_probabilities = tl.math.exp2(current_scores - new_max)
            current_probabilities = tl.where(
                current_mask, current_probabilities, 0.0
            )
            normalizer = normalizer * old_factor + current_probabilities
            current_value_row = tl.load(
                current_value_ptr + current_value_base + dimensions,
                mask=current_active & dimension_mask,
                other=0.0,
            ).to(tl.bfloat16)
            accumulator = accumulator * old_factor[:, None] + (
                current_probabilities[:, None]
                * current_value_row[None, :].to(tl.float32)
            )
            max_score = new_max

            current_ring_base = (
                state_slot * stride_hot_slot
                + kv_head * stride_hot_head
                + ((sequence_length - 1) % hot_capacity) * stride_hot_token
            )
            owner_mask = (
                (state_slot >= 0)
                & (group_index == 0)
                & (split_index == current_owner)
            )
            tl.store(
                hot_key_ptr + current_ring_base + dimensions,
                current_key_row,
                mask=owner_mask & dimension_mask,
            )
            tl.store(
                hot_value_ptr + current_ring_base + dimensions,
                current_value_row,
                mask=owner_mask & dimension_mask,
            )

        result = accumulator / tl.maximum(normalizer[:, None], 1e-20)
        if num_splits == 1:
            output_bases = (
                request_index * stride_output_token
                + query_heads[:, None] * stride_output_head
            )
            tl.store(
                output_ptr + output_bases + dimensions[None, :],
                result,
                mask=group_mask[:, None] & dimension_mask[None, :],
            )
        else:
            mid_bases = (
                request_index * stride_mid_batch
                + query_heads[:, None] * stride_mid_head
                + split_index * stride_mid_split
            )
            tl.store(
                mid_o_ptr + mid_bases + dimensions[None, :],
                result,
                mask=group_mask[:, None] & dimension_mask[None, :],
            )
            local_lse = tl.where(
                normalizer > 0.0,
                max_score + tl.math.log2(normalizer),
                -float("inf"),
            )
            tl.store(
                mid_o_ptr
                + request_index * stride_mid_batch
                + query_heads * stride_mid_head
                + split_index * stride_mid_split
                + head_size,
                local_lse,
                mask=group_mask,
            )

    @triton.jit
    def _hyquant_grouped_split_reduce_kernel(
        mid_o_ptr,
        output_ptr,
        stride_mid_batch,
        stride_mid_head,
        stride_mid_split,
        stride_output_token,
        stride_output_head,
        head_size: tl.constexpr,
        num_splits: tl.constexpr,
        block_d: tl.constexpr,
    ):
        request_index = tl.program_id(0)
        query_head = tl.program_id(1)
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < head_size
        base = request_index * stride_mid_batch + query_head * stride_mid_head
        accumulator = tl.zeros([block_d], tl.float32)
        max_lse = tl.full([], -float("inf"), tl.float32)
        normalizer = tl.zeros([], tl.float32)
        for split_index in range(num_splits):
            partial = tl.load(
                mid_o_ptr + base + split_index * stride_mid_split + dimensions,
                mask=dimension_mask,
                other=0.0,
            ).to(tl.float32)
            partial_lse = tl.load(
                mid_o_ptr + base + split_index * stride_mid_split + head_size
            ).to(tl.float32)
            new_max = tl.maximum(max_lse, partial_lse)
            old_factor = tl.where(
                normalizer > 0.0,
                tl.math.exp2(max_lse - new_max),
                0.0,
            )
            partial_factor = tl.where(
                partial_lse > -float("inf"),
                tl.math.exp2(partial_lse - new_max),
                0.0,
            )
            accumulator = accumulator * old_factor + partial * partial_factor
            normalizer = normalizer * old_factor + partial_factor
            max_lse = new_max
        output_base = (
            request_index * stride_output_token
            + query_head * stride_output_head
            + dimensions
        )
        tl.store(
            output_ptr + output_base,
            accumulator / tl.maximum(normalizer, 1e-20),
            mask=dimension_mask,
        )


def triton_hyquant_grouped_decode_attention(
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
    """Launch the grouped compact-page decode kernel when the shape is valid."""
    global _LAST_STATUS
    if not HAS_TRITON or not query.is_cuda:
        _LAST_STATUS = "no_triton_or_cuda"
        return None
    if query.ndim != 3 or query.shape[0] != seq_lens.shape[0]:
        _LAST_STATUS = "shape"
        return None
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 3
        or kv_cache.shape[1] <= 0
        or kv_cache.shape[2] != layout.head_page_bytes
        or query.shape[1] % kv_cache.shape[1]
    ):
        _LAST_STATUS = "cache_shape"
        return None
    if kv_cache.stride(-1) != 1 or layout.head_page_bytes % 4:
        _LAST_STATUS = "cache_alignment"
        return None
    if query.stride(-1) != 1 or any(stride < 0 for stride in query.stride()):
        _LAST_STATUS = "query_stride"
        return None
    if output is None:
        output = torch.empty_like(query)
    elif (
        output.shape != query.shape
        or output.device != query.device
        or output.stride(-1) != 1
    ):
        _LAST_STATUS = "output_shape"
        return None
    if hot_key.shape != hot_value.shape or hot_key.dtype != torch.bfloat16:
        _LAST_STATUS = "hot_shape"
        return None
    if hot_key.ndim != 4 or hot_key.shape[1] != kv_cache.shape[1]:
        _LAST_STATUS = "hot_shape"
        return None
    if fuse_current_store:
        expected_current_shape = (
            query.shape[0],
            kv_cache.shape[1],
            query.shape[-1],
        )
        if current_key is None or current_value is None:
            _LAST_STATUS = "missing_current_kv"
            return None
        if (
            current_key.shape != expected_current_shape
            or current_value.shape != expected_current_shape
            or current_key.device != query.device
            or current_value.device != query.device
            or current_key.dtype != torch.bfloat16
            or current_value.dtype != torch.bfloat16
            or current_key.stride(-1) != 1
            or current_value.stride(-1) != 1
        ):
            _LAST_STATUS = "current_kv_shape"
            return None
    else:
        # Keep the launch ABI uniform.  constexpr folding removes all accesses
        # to these aliases when fusion is disabled.
        current_key = query
        current_value = query
    if layout.block_size not in (16, 32, 64) or layout.head_size > 256:
        _LAST_STATUS = "unsupported_shape"
        return None

    batch_size = int(query.shape[0])
    query_group_size = query.shape[1] // kv_cache.shape[1]
    if max_seq_len is None:
        max_seq_len = max(1, int(seq_lens.max().item()))
    else:
        max_seq_len = max(1, int(max_seq_len))
    num_splits = get_hyquant_decode_split_count(
        batch_size, max_seq_len, query_group_size
    )
    block_kv, block_h, num_warps = get_hyquant_decode_launch_config(
        batch_size, max_seq_len, query_group_size
    )
    block_d = 1 << (query.shape[-1] - 1).bit_length()
    block_half = 1 << (layout.packed_value_bytes - 1).bit_length()
    block_packed_half = 1 << ((layout.packed_value_bytes // 2) - 1).bit_length()
    block_g = 1 << (layout.num_groups - 1).bit_length()
    quant_per_block = max(1, layout.block_size - layout.anchor_count)
    block_table_tile = 1 << (
        (block_kv + quant_per_block - 1) // quant_per_block - 1
    ).bit_length()
    group_tiles = (query_group_size + block_h - 1) // block_h

    if num_splits > 1:
        required_shape = (
            batch_size,
            query.shape[1],
            num_splits,
            block_d + 1,
        )
        if (
            mid_o_buf is None
            or mid_o_buf.dtype != torch.float32
            or mid_o_buf.device != query.device
            or mid_o_buf.ndim != 4
            or any(
                actual < required
                for actual, required in zip(mid_o_buf.shape, required_shape)
            )
        ):
            if torch.cuda.is_current_stream_capturing():
                _LAST_STATUS = "missing_cudagraph_workspace"
                return None
            mid_o = torch.empty(
                required_shape, dtype=torch.float32, device=query.device
            )
        else:
            mid_o = mid_o_buf[
                :batch_size, : query.shape[1], :num_splits, : block_d + 1
            ]
    else:
        # The pointer is still passed to keep the launch signature uniform.
        mid_o = kv_cache

    cache_i16 = kv_cache.view(torch.int16)
    coalesce_block_table = int(
        os.environ.get("VLLM_HYQUANT_DECODE_COALESCE_TABLE", "1") != "0"
    )
    _hyquant_grouped_split_decode_kernel[
        (batch_size, kv_cache.shape[1], group_tiles * num_splits)
    ](
        query,
        current_key,
        current_value,
        kv_cache,
        cache_i16,
        hot_key,
        hot_value,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        output,
        mid_o,
        query.stride(0),
        query.stride(1),
        current_key.stride(0),
        current_key.stride(1),
        current_value.stride(0),
        current_value.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        hot_key.stride(0),
        hot_key.stride(1),
        hot_key.stride(2),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
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
        block_h=block_h,
        block_half=block_half,
        block_packed_half=block_packed_half,
        block_g=block_g,
        block_d=block_d,
        block_kv=block_kv,
        block_table_tile=block_table_tile,
        num_splits=num_splits,
        coalesce_block_table=coalesce_block_table,
        fuse_current_store=int(fuse_current_store),
        softmax_scale2=scale * _LOG2_E,
        num_warps=num_warps,
        num_stages=int(
            os.environ.get(
                "VLLM_HYQUANT_DECODE_STAGES",
                "1" if (batch_size >= 4 or max_seq_len >= 2048) else "2",
            )
        ),
    )
    if num_splits > 1:
        _hyquant_grouped_split_reduce_kernel[(batch_size, query.shape[1])] (
            mid_o,
            output,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            output.stride(0),
            output.stride(1),
            head_size=query.shape[-1],
            num_splits=num_splits,
            block_d=block_d,
            num_warps=4,
            num_stages=1,
        )
        _LAST_STATUS = "grouped_splitk"
    else:
        _LAST_STATUS = "grouped"
    return output
