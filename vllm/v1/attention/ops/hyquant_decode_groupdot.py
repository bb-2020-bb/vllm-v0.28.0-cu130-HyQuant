"""Fast grouped decode for the compact HyQuant page format.

Quantized rows stay packed in the page.  The decoder unpacks each channel
group just in time for a BF16 tensor-core dot and applies the per-token,
per-group scale to the dot result.  This avoids materialising a full BF16
``[tokens, head_size]`` tile, which was the dominant cost of the former
decoder.
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
    """Choose a split count tuned for the SM80 grouped decode shape."""
    override = os.environ.get("VLLM_HYQUANT_DECODE_SPLITS")
    if override:
        value = int(override)
        if value not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError(
                "VLLM_HYQUANT_DECODE_SPLITS must be one of 1,2,4,8,16,32,64"
            )
        return value
    if batch_size <= 0 or query_group_size <= 1 or max_seq_len < 512:
        return 1
    # On SM80 a single long GQA request is occupancy limited.  64-way
    # splitting gives enough independent CTAs to hide the packed-cache
    # latency; for shorter contexts the extra reduction CTAs cost more than
    # they save.  Keep the policy conservative for larger batches where the
    # request dimension already supplies occupancy.
    if batch_size == 1:
        return 64 if max_seq_len >= 8192 else (16 if max_seq_len >= 2048 else 4)
    if batch_size == 2:
        return (
            64
            if max_seq_len >= 16384
            else (32 if max_seq_len >= 8192 else (16 if max_seq_len >= 2048 else 4))
        )
    if batch_size <= 8:
        return (
            32
            if max_seq_len >= 8192
            else (16 if max_seq_len >= 4096 else (8 if max_seq_len >= 2048 else 4))
        )
    if batch_size <= 16:
        return 16 if max_seq_len >= 6144 else 8
    return 16 if max_seq_len >= 4096 else 8


def get_hyquant_decode_launch_config(
    batch_size: int, max_seq_len: int, query_group_size: int
) -> tuple[int, int, int]:
    """Return ``(BLOCK_KV, BLOCK_H, num_warps)`` for the decoder."""
    block_kv = (
        128
        if batch_size <= 8 and max_seq_len >= 2048
        else 64
        if batch_size <= 8
        else 32
    )
    if query_group_size <= 1:
        block_h = 1
    elif query_group_size <= 4:
        block_h = query_group_size
    else:
        block_h = 4
    block_kv_override = os.environ.get("VLLM_HYQUANT_DECODE_BLOCK_KV")
    if block_kv_override:
        block_kv = int(block_kv_override)
        if block_kv not in (32, 64, 128, 256):
            raise ValueError("VLLM_HYQUANT_DECODE_BLOCK_KV must be 32/64/128/256")
    block_h_override = os.environ.get("VLLM_HYQUANT_DECODE_BLOCK_H")
    if block_h_override:
        block_h = int(block_h_override)
        if block_h not in (1, 2, 4, 8):
            raise ValueError("VLLM_HYQUANT_DECODE_BLOCK_H must be 1/2/4/8")
    num_warps = int(os.environ.get("VLLM_HYQUANT_DECODE_WARPS", "4"))
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("VLLM_HYQUANT_DECODE_WARPS must be 1/2/4/8")
    return block_kv, block_h, num_warps


def get_hyquant_decode_num_stages(batch_size: int, max_seq_len: int) -> int:
    """Choose the software-pipeline depth for packed-cache loads.

    Three stages are useful for the long, low-batch SM80 case.  Larger
    batches have enough concurrent CTAs that the additional registers reduce
    occupancy, so they retain the two-stage/default path.
    """
    override = os.environ.get("VLLM_HYQUANT_DECODE_STAGES")
    if override:
        value = int(override)
        if value not in (1, 2, 3, 4):
            raise ValueError("VLLM_HYQUANT_DECODE_STAGES must be 1/2/3/4")
        return value
    if batch_size <= 2 and max_seq_len >= 8192:
        return 3
    return 2


if HAS_TRITON:

    @triton.jit
    def _hyquant_groupdot_split_decode_kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
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
        stride_mid_batch,
        stride_mid_head,
        stride_mid_split,
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
        block_h: tl.constexpr,
        block_d: tl.constexpr,
        block_kv: tl.constexpr,
        block_words: tl.constexpr,
        block_table_tile: tl.constexpr,
        coalesce_block_table: tl.constexpr,
        coalesce_scales: tl.constexpr,
        num_splits: tl.constexpr,
        use_simt: tl.constexpr,
        int8_qk: tl.constexpr,
        fuse_current_store: tl.constexpr,
        softmax_scale2: tl.constexpr,
    ):
        request_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        group_split = tl.program_id(2)
        split_index = group_split % num_splits
        group_index = group_split // num_splits

        h_offsets = tl.arange(0, block_h)
        group_start = group_index * block_h
        query_heads = kv_head * query_group_size + group_start + h_offsets
        h_mask = h_offsets < (query_group_size - group_start)
        d_offsets = tl.arange(0, group_size)
        d_mask = d_offsets < group_size
        kv_offsets = tl.arange(0, block_kv)
        word_offsets = tl.arange(0, block_words)
        # Four is the maximum supported group count.  Loading all group
        # scales as one contiguous tile removes a strided memory instruction
        # for every group; the compile-time guard keeps the generic 1/2/3
        # group layouts on their original ABI-safe path.
        scale_offsets = tl.arange(0, 4)

        # Keep the query in small group tiles.  Quantized QK/PV dots consume
        # these tiles directly, so no full BF16 quantized-K/V tile is formed.
        query_bases = (
            request_index * stride_query_token
            + query_heads[:, None] * stride_query_head
        )
        q0 = tl.load(
            query_ptr + query_bases + d_offsets[None, :],
            mask=h_mask[:, None] & d_mask[None, :] & ((d_offsets)[None, :] < head_size),
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)
        if num_groups > 1:
            q1 = tl.load(
                query_ptr + query_bases + group_size + d_offsets[None, :],
                mask=h_mask[:, None]
                & d_mask[None, :]
                & ((group_size + d_offsets)[None, :] < head_size),
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.bfloat16)
        else:
            q1 = tl.zeros([block_h, group_size], tl.bfloat16)
        if num_groups > 2:
            q2 = tl.load(
                query_ptr + query_bases + 2 * group_size + d_offsets[None, :],
                mask=h_mask[:, None]
                & d_mask[None, :]
                & ((2 * group_size + d_offsets)[None, :] < head_size),
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.bfloat16)
        else:
            q2 = tl.zeros([block_h, group_size], tl.bfloat16)
        if num_groups > 3:
            q3 = tl.load(
                query_ptr + query_bases + 3 * group_size + d_offsets[None, :],
                mask=h_mask[:, None]
                & d_mask[None, :]
                & ((3 * group_size + d_offsets)[None, :] < head_size),
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.bfloat16)
        else:
            q3 = tl.zeros([block_h, group_size], tl.bfloat16)

        # On SM80 the INT8 dot path can evaluate QK directly from the packed
        # signed INT4 codes.  Quantize each query group once per CTA; the
        # resulting scale is folded into the per-row K scale below.  Keep this
        # behind a constexpr flag so the normal BF16 path has no extra code or
        # registers when the integer candidate is disabled.
        if int8_qk:
            q0_f = q0.to(tl.float32)
            q1_f = q1.to(tl.float32)
            q2_f = q2.to(tl.float32)
            q3_f = q3.to(tl.float32)
            qscale0 = tl.maximum(tl.max(tl.abs(q0_f), axis=1) / 127.0, 1e-8)
            qscale1 = tl.maximum(tl.max(tl.abs(q1_f), axis=1) / 127.0, 1e-8)
            qscale2 = tl.maximum(tl.max(tl.abs(q2_f), axis=1) / 127.0, 1e-8)
            qscale3 = tl.maximum(tl.max(tl.abs(q3_f), axis=1) / 127.0, 1e-8)
            q0_i8 = tl.where(
                q0_f >= 0,
                q0_f / qscale0[:, None] + 0.5,
                q0_f / qscale0[:, None] - 0.5,
            ).to(tl.int8)
            q1_i8 = tl.where(
                q1_f >= 0,
                q1_f / qscale1[:, None] + 0.5,
                q1_f / qscale1[:, None] - 0.5,
            ).to(tl.int8)
            q2_i8 = tl.where(
                q2_f >= 0,
                q2_f / qscale2[:, None] + 0.5,
                q2_f / qscale2[:, None] - 0.5,
            ).to(tl.int8)
            q3_i8 = tl.where(
                q3_f >= 0,
                q3_f / qscale3[:, None] + 0.5,
                q3_f / qscale3[:, None] - 0.5,
            ).to(tl.int8)
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
        table_base = request_index * stride_block_table

        max_score = tl.full([block_h], -float("inf"), tl.float32)
        normalizer = tl.zeros([block_h], tl.float32)
        acc0 = tl.zeros([block_h, group_size], tl.float32)
        acc1 = tl.zeros([block_h, group_size], tl.float32)
        acc2 = tl.zeros([block_h, group_size], tl.float32)
        acc3 = tl.zeros([block_h, group_size], tl.float32)

        # Quantized rows are contiguous in every compact page.  Guarding this
        # block at compile time also keeps the all-anchor configuration from
        # generating a divide-by-zero expression.
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
                    # A BLOCK_KV tile spans only a handful of compact pages.
                    # Read that short page-table range once and gather the
                    # corresponding entry for each token.  This avoids a
                    # duplicate random page-table read for every row.
                    first_block = start // quant_per_block
                    table_offsets = tl.arange(0, block_table_tile)
                    table_valid = (
                        table_offsets
                        < tl.cdiv(
                            (start % quant_per_block) + block_kv,
                            quant_per_block,
                        )
                    ) & (first_block + table_offsets < committed_blocks)
                    table_values = tl.load(
                        block_table_ptr + table_base + first_block + table_offsets,
                        mask=table_valid,
                        other=0,
                    ).to(tl.int64)
                    safe_select = tl.minimum(
                        tl.maximum(logical_blocks - first_block, 0),
                        block_table_tile - 1,
                    )
                    physical_blocks = tl.gather(table_values, safe_select, axis=0)
                else:
                    safe_blocks = tl.where(valid, logical_blocks, 0)
                    physical_blocks = tl.load(
                        block_table_ptr + table_base + safe_blocks,
                        mask=valid,
                        other=0,
                    ).to(tl.int64)
                page_bases = (
                    physical_blocks * stride_cache_block + kv_head * stride_cache_head
                )

                if coalesce_scales:
                    key_scale_bits_all = tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + key_scale_offset) // 2
                        + quant_rows[:, None] * num_groups
                        + scale_offsets[None, :],
                        mask=valid[:, None] & (scale_offsets[None, :] < num_groups),
                        other=0,
                        cache_modifier=".cg",
                    )
                    value_scale_bits_all = tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + value_scale_offset) // 2
                        + quant_rows[:, None] * num_groups
                        + scale_offsets[None, :],
                        mask=valid[:, None] & (scale_offsets[None, :] < num_groups),
                        other=0,
                        cache_modifier=".cg",
                    )
                    key_scales_all = key_scale_bits_all.to(tl.float16, bitcast=True).to(
                        tl.float32
                    )
                    value_scales_all = value_scale_bits_all.to(
                        tl.float16, bitcast=True
                    ).to(tl.float32)
                    # ``tl.split`` is used instead of tensor slicing because
                    # slicing a constexpr-rank Triton tensor is not supported
                    # consistently across the Triton releases used by vLLM.
                    key_scale_02, key_scale_13 = tl.split(
                        tl.reshape(key_scales_all, (block_kv, 2, 2))
                    )
                    key_scale_0, key_scale_2 = tl.split(key_scale_02)
                    key_scale_1, key_scale_3 = tl.split(key_scale_13)
                    value_scale_02, value_scale_13 = tl.split(
                        tl.reshape(value_scales_all, (block_kv, 2, 2))
                    )
                    value_scale_0, value_scale_2 = tl.split(value_scale_02)
                    value_scale_1, value_scale_3 = tl.split(value_scale_13)

                # QK: independent group MMA operations over raw signed INT4
                # codes.  The per-token/group scale is applied to the dot
                # result rather than expanding a BF16 tile.
                scores = tl.zeros([block_h, block_kv], tl.float32)
                for g in range(num_groups):
                    packed = (
                        tl.load(
                            cache_i16_ptr
                            + (page_bases[:, None] + quant_key_offset) // 2
                            + quant_rows[:, None] * (packed_value_bytes // 2)
                            + g * block_words
                            + word_offsets[None, :],
                            mask=valid[:, None] & (word_offsets[None, :] < block_words),
                            other=0,
                            cache_modifier=".cg",
                        ).to(tl.int32)
                        & 0xFFFF
                    )
                    n0 = ((packed & 0xF) ^ 0x8) - 0x8
                    n1 = (((packed >> 4) & 0xF) ^ 0x8) - 0x8
                    n2 = (((packed >> 8) & 0xF) ^ 0x8) - 0x8
                    n3 = (((packed >> 12) & 0xF) ^ 0x8) - 0x8
                    if int8_qk:
                        codes = tl.reshape(
                            tl.join(tl.join(n0, n2), tl.join(n1, n3)),
                            (block_kv, group_size),
                        ).to(tl.int8)
                        if g == 0:
                            qg = q0_i8
                            qscale = qscale0
                        elif g == 1:
                            qg = q1_i8
                            qscale = qscale1
                        elif g == 2:
                            qg = q2_i8
                            qscale = qscale2
                        else:
                            qg = q3_i8
                            qscale = qscale3
                        raw_score = tl.dot(qg, tl.trans(codes), out_dtype=tl.int32).to(
                            tl.float32
                        )
                    else:
                        codes = tl.reshape(
                            tl.join(tl.join(n0, n2), tl.join(n1, n3)),
                            (block_kv, group_size),
                        ).to(tl.bfloat16)
                        if g == 0:
                            qg = q0
                        elif g == 1:
                            qg = q1
                        elif g == 2:
                            qg = q2
                        else:
                            qg = q3
                        if use_simt:
                            raw_score = tl.sum(
                                qg.to(tl.float32)[:, None, :]
                                * codes.to(tl.float32)[None, :, :],
                                axis=2,
                            )
                        else:
                            raw_score = tl.dot(
                                qg, tl.trans(codes), input_precision="bf16x3"
                            ).to(tl.float32)
                    if coalesce_scales:
                        if g == 0:
                            key_scale = key_scale_0
                        elif g == 1:
                            key_scale = key_scale_1
                        elif g == 2:
                            key_scale = key_scale_2
                        else:
                            key_scale = key_scale_3
                    else:
                        scale_bits = tl.load(
                            cache_i16_ptr
                            + (page_bases + key_scale_offset) // 2
                            + quant_rows * num_groups
                            + g,
                            mask=valid,
                            other=0,
                            cache_modifier=".cg",
                        )
                        key_scale = scale_bits.to(tl.float16, bitcast=True).to(
                            tl.float32
                        )
                    if int8_qk:
                        scores += raw_score * (qscale[:, None] * key_scale[None, :])
                    else:
                        scores += raw_score * key_scale[None, :]

                score_mask = h_mask[:, None] & valid[None, :]
                scores = scores * softmax_scale2
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

                # PV: scale P by the per-row V scale before the MMA, avoiding
                # materialisation of dequantized BF16 values.
                for g in range(num_groups):
                    packed = (
                        tl.load(
                            cache_i16_ptr
                            + (page_bases[:, None] + quant_value_offset) // 2
                            + quant_rows[:, None] * (packed_value_bytes // 2)
                            + g * block_words
                            + word_offsets[None, :],
                            mask=valid[:, None] & (word_offsets[None, :] < block_words),
                            other=0,
                            cache_modifier=".cg",
                        ).to(tl.int32)
                        & 0xFFFF
                    )
                    n0 = ((packed & 0xF) ^ 0x8) - 0x8
                    n1 = (((packed >> 4) & 0xF) ^ 0x8) - 0x8
                    n2 = (((packed >> 8) & 0xF) ^ 0x8) - 0x8
                    n3 = (((packed >> 12) & 0xF) ^ 0x8) - 0x8
                    codes = tl.reshape(
                        tl.join(tl.join(n0, n2), tl.join(n1, n3)),
                        (block_kv, group_size),
                    ).to(tl.bfloat16)
                    if coalesce_scales:
                        if g == 0:
                            value_scale = value_scale_0
                        elif g == 1:
                            value_scale = value_scale_1
                        elif g == 2:
                            value_scale = value_scale_2
                        else:
                            value_scale = value_scale_3
                    else:
                        scale_bits = tl.load(
                            cache_i16_ptr
                            + (page_bases + value_scale_offset) // 2
                            + quant_rows * num_groups
                            + g,
                            mask=valid,
                            other=0,
                            cache_modifier=".cg",
                        )
                        value_scale = scale_bits.to(tl.float16, bitcast=True).to(
                            tl.float32
                        )
                    scaled_p = (probabilities * value_scale[None, :]).to(tl.bfloat16)
                    if use_simt:
                        update = tl.sum(
                            scaled_p.to(tl.float32)[:, :, None]
                            * codes.to(tl.float32)[None, :, :],
                            axis=1,
                        )
                    else:
                        update = tl.dot(scaled_p, codes, input_precision="bf16x3").to(
                            tl.float32
                        )
                    if g == 0:
                        acc0 = acc0 * old_factor[:, None] + update
                    elif g == 1:
                        acc1 = acc1 * old_factor[:, None] + update
                    elif g == 2:
                        acc2 = acc2 * old_factor[:, None] + update
                    else:
                        acc3 = acc3 * old_factor[:, None] + update
                max_score = new_max

        # The sparse anchor rows are few (one per page), so retain a simple
        # BF16 path for them.  They share the same online softmax state.
        anchor_total = committed_blocks * anchor_count
        anchor_tiles = tl.cdiv(anchor_total, block_kv)
        anchor_tiles_per_split = tl.cdiv(anchor_tiles, num_splits)
        anchor_start = split_index * anchor_tiles_per_split * block_kv
        anchor_end = tl.minimum(
            anchor_start + anchor_tiles_per_split * block_kv, anchor_total
        )
        for start in tl.range(anchor_start, anchor_end, block_kv):
            indices = start + kv_offsets
            valid = indices < anchor_total
            blocks = indices // anchor_count
            rows = indices % anchor_count
            safe_blocks = tl.where(valid, blocks, 0)
            physical = tl.load(
                block_table_ptr + table_base + safe_blocks,
                mask=valid,
                other=0,
            ).to(tl.int64)
            pages = physical * stride_cache_block + kv_head * stride_cache_head
            key_base = (pages + anchor_key_offset) // 2 + rows * head_size
            # Build the full-D score from group tiles without relying on
            # tensor slicing (unsupported by some Triton releases).
            scores = tl.zeros([block_h, block_kv], tl.float32)
            for g in range(num_groups):
                kg = tl.load(
                    cache_i16_ptr
                    + key_base[:, None]
                    + g * group_size
                    + d_offsets[None, :],
                    mask=valid[:, None]
                    & d_mask[None, :]
                    & ((g * group_size + d_offsets)[None, :] < head_size),
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.bfloat16, bitcast=True)
                if g == 0:
                    qg = q0
                elif g == 1:
                    qg = q1
                elif g == 2:
                    qg = q2
                else:
                    qg = q3
                if use_simt:
                    scores += tl.sum(
                        qg.to(tl.float32)[:, None, :] * kg.to(tl.float32)[None, :, :],
                        axis=2,
                    )
                else:
                    scores += tl.dot(qg, tl.trans(kg), input_precision="bf16x3").to(
                        tl.float32
                    )
            score_mask = h_mask[:, None] & valid[None, :]
            scores = tl.where(score_mask, scores * softmax_scale2, -float("inf"))
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
            value_base = (pages + anchor_value_offset) // 2 + rows * head_size
            for g in range(num_groups):
                vg = tl.load(
                    cache_i16_ptr
                    + value_base[:, None]
                    + g * group_size
                    + d_offsets[None, :],
                    mask=valid[:, None]
                    & d_mask[None, :]
                    & ((g * group_size + d_offsets)[None, :] < head_size),
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.bfloat16, bitcast=True)
                if use_simt:
                    anchor_update = tl.sum(
                        probabilities.to(tl.float32)[:, :, None]
                        * vg.to(tl.float32)[None, :, :],
                        axis=1,
                    )
                else:
                    anchor_update = tl.dot(
                        probabilities.to(tl.bfloat16),
                        vg,
                        input_precision="bf16x3",
                    ).to(tl.float32)
                if g == 0:
                    acc0 = acc0 * old_factor[:, None] + anchor_update
                elif g == 1:
                    acc1 = acc1 * old_factor[:, None] + anchor_update
                elif g == 2:
                    acc2 = acc2 * old_factor[:, None] + anchor_update
                else:
                    acc3 = acc3 * old_factor[:, None] + anchor_update
            max_score = new_max

        # Recent BF16 rows live in the ring.  In fused mode the current row is
        # consumed below and must be excluded here; otherwise it would be
        # counted twice after the append.
        hot_count = sequence_length - committed
        hot_scan_count = (
            tl.maximum(hot_count - 1, 0) if fuse_current_store else hot_count
        )
        hot_tiles = tl.cdiv(hot_scan_count, block_kv)
        hot_tiles_per_split = tl.cdiv(hot_tiles, num_splits)
        hot_start = split_index * hot_tiles_per_split * block_kv
        hot_end = tl.minimum(hot_start + hot_tiles_per_split * block_kv, hot_scan_count)
        for start in tl.range(hot_start, hot_end, block_kv):
            indices = start + kv_offsets
            valid = indices < hot_scan_count
            logical = committed + indices
            ring = logical % hot_capacity
            bases = (
                state_slot * stride_hot_slot
                + kv_head * stride_hot_head
                + ring[:, None] * stride_hot_token
            )
            scores = tl.zeros([block_h, block_kv], tl.float32)
            for g in range(num_groups):
                kg = tl.load(
                    hot_key_ptr + bases + g * group_size + d_offsets[None, :],
                    mask=valid[:, None]
                    & d_mask[None, :]
                    & ((g * group_size + d_offsets)[None, :] < head_size),
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.bfloat16)
                if g == 0:
                    qg = q0
                elif g == 1:
                    qg = q1
                elif g == 2:
                    qg = q2
                else:
                    qg = q3
                if use_simt:
                    scores += tl.sum(
                        qg.to(tl.float32)[:, None, :] * kg.to(tl.float32)[None, :, :],
                        axis=2,
                    )
                else:
                    scores += tl.dot(qg, tl.trans(kg), input_precision="bf16x3").to(
                        tl.float32
                    )
            score_mask = h_mask[:, None] & valid[None, :]
            scores = tl.where(score_mask, scores * softmax_scale2, -float("inf"))
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
            for g in range(num_groups):
                vg = tl.load(
                    hot_value_ptr + bases + g * group_size + d_offsets[None, :],
                    mask=valid[:, None]
                    & d_mask[None, :]
                    & ((g * group_size + d_offsets)[None, :] < head_size),
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.bfloat16)
                if use_simt:
                    hot_update = tl.sum(
                        probabilities.to(tl.float32)[:, :, None]
                        * vg.to(tl.float32)[None, :, :],
                        axis=1,
                    )
                else:
                    hot_update = tl.dot(
                        probabilities.to(tl.bfloat16),
                        vg,
                        input_precision="bf16x3",
                    ).to(tl.float32)
                if g == 0:
                    acc0 = acc0 * old_factor[:, None] + hot_update
                elif g == 1:
                    acc1 = acc1 * old_factor[:, None] + hot_update
                elif g == 2:
                    acc2 = acc2 * old_factor[:, None] + hot_update
                else:
                    acc3 = acc3 * old_factor[:, None] + hot_update
            max_score = new_max

        if fuse_current_store:
            # Only the split owning the final hot tile consumes and publishes
            # the current row.  This keeps split-K reduction exact while
            # avoiding a separate hot-store launch on every decode step.
            current_owner = tl.minimum(
                tl.maximum(hot_count - 1, 0)
                // tl.maximum(hot_tiles_per_split * block_kv, 1),
                num_splits - 1,
            )
            current_key_base = (
                request_index * stride_current_key_token
                + kv_head * stride_current_key_head
            )
            current_value_base = (
                request_index * stride_current_value_token
                + kv_head * stride_current_value_head
            )
            current_active = (state_slot >= 0) & (split_index == current_owner)
            # Reuse the group-sized query tiles already resident in registers.
            # The previous fused path loaded a second full-D query row just to
            # score this one token; that load is larger than the current KV
            # row and inflated register pressure.  A groupwise dot is exact
            # for the BF16 current row and keeps the fused path aligned with
            # the packed QK computation.
            current_scores = tl.zeros([block_h], tl.float32)
            current_ring_base = (
                state_slot * stride_hot_slot
                + kv_head * stride_hot_head
                + ((sequence_length - 1) % hot_capacity) * stride_hot_token
            )
            owner_mask = (
                (state_slot >= 0) & (group_index == 0) & (split_index == current_owner)
            )
            for g in range(num_groups):
                if g == 0:
                    qg = q0
                elif g == 1:
                    qg = q1
                elif g == 2:
                    qg = q2
                else:
                    qg = q3
                current_key_group = tl.load(
                    current_key_ptr + current_key_base + g * group_size + d_offsets,
                    mask=current_active
                    & d_mask
                    & ((g * group_size + d_offsets) < head_size),
                    other=0.0,
                ).to(tl.bfloat16)
                current_scores += tl.sum(
                    qg.to(tl.float32) * current_key_group[None, :].to(tl.float32),
                    axis=1,
                )
                # The owner CTA can publish the key while it is already in
                # registers for QK.  This removes the second full-row global
                # load that the former fused tail performed.
                tl.store(
                    hot_key_ptr + current_ring_base + g * group_size + d_offsets,
                    current_key_group,
                    mask=owner_mask
                    & d_mask
                    & ((g * group_size + d_offsets) < head_size),
                )
            current_scores *= softmax_scale2
            current_mask = h_mask & (state_slot >= 0) & (split_index == current_owner)
            current_scores = tl.where(current_mask, current_scores, -float("inf"))
            new_max = tl.maximum(max_score, current_scores)
            old_factor = tl.where(
                new_max > -float("inf"),
                tl.math.exp2(max_score - new_max),
                0.0,
            )
            current_probabilities = tl.math.exp2(current_scores - new_max)
            current_probabilities = tl.where(current_mask, current_probabilities, 0.0)
            normalizer = normalizer * old_factor + current_probabilities
            # Keep the current contribution in the same per-group
            # accumulators as the packed/anchor/window rows.  A separate full-D
            # accumulator would be lost at the split-K merge and was the source
            # of an earlier fused-path correctness bug.
            for g in range(num_groups):
                current_value_group = tl.load(
                    current_value_ptr + current_value_base + g * group_size + d_offsets,
                    mask=current_active
                    & d_mask
                    & ((g * group_size + d_offsets) < head_size),
                    other=0.0,
                ).to(tl.bfloat16)
                current_update = current_probabilities[:, None] * (
                    current_value_group[None, :].to(tl.float32)
                )
                # Likewise publish V directly from the value contribution
                # load.  Only the owner CTA stores, so no duplicate writers
                # are introduced across query-head groups or split-K CTAs.
                tl.store(
                    hot_value_ptr + current_ring_base + g * group_size + d_offsets,
                    current_value_group,
                    mask=owner_mask
                    & d_mask
                    & ((g * group_size + d_offsets) < head_size),
                )
                if g == 0:
                    acc0 = acc0 * old_factor[:, None] + current_update
                elif g == 1:
                    acc1 = acc1 * old_factor[:, None] + current_update
                elif g == 2:
                    acc2 = acc2 * old_factor[:, None] + current_update
                else:
                    acc3 = acc3 * old_factor[:, None] + current_update
            max_score = new_max

        # Write each group directly.  This avoids constructing a temporary
        # [BLOCK_H, BLOCK_D] tensor and works for both the direct and split-K
        # output paths.
        inv_norm = 1.0 / tl.maximum(normalizer[:, None], 1e-20)
        for g in range(num_groups):
            if g == 0:
                acc_out = acc0
            elif g == 1:
                acc_out = acc1
            elif g == 2:
                acc_out = acc2
            else:
                acc_out = acc3
            if num_splits == 1:
                result_bases = (
                    request_index * stride_output_token
                    + query_heads[:, None] * stride_output_head
                    + g * group_size
                )
                tl.store(
                    output_ptr + result_bases + d_offsets[None, :],
                    acc_out * inv_norm,
                    mask=h_mask[:, None]
                    & d_mask[None, :]
                    & ((g * group_size + d_offsets)[None, :] < head_size),
                )
            else:
                # Store normalized partials and one log-normalizer for the
                # lightweight split-K merge kernel.
                result_bases = (
                    request_index * stride_mid_batch
                    + query_heads[:, None] * stride_mid_head
                    + split_index * stride_mid_split
                    + g * group_size
                )
                tl.store(
                    mid_o_ptr + result_bases + d_offsets[None, :],
                    acc_out * inv_norm,
                    mask=h_mask[:, None]
                    & d_mask[None, :]
                    & ((g * group_size + d_offsets)[None, :] < head_size),
                )
        if num_splits > 1:
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
                mask=h_mask,
            )


if HAS_TRITON:

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
                mid_o_ptr + base + split_index * stride_mid_split + head_size,
                mask=True,
                other=-float("inf"),
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


def triton_hyquant_groupdot_decode_attention(
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
    """Launch the direct INT4 group-dot decoder.

    The function deliberately returns ``None`` for unsupported layouts instead
    of silently changing the cache ABI.  The caller can then select the
    compatibility decoder.  Compilation errors are allowed to propagate so a
    strict deployment can detect a target-specific Triton failure.
    """
    global _LAST_STATUS
    if not HAS_TRITON or not query.is_cuda:
        _LAST_STATUS = "no_triton_or_cuda"
        return None

    # Fast-path the first production shape before the generic validation and
    # dispatch bookkeeping below.  This function is called once per decoder
    # layer; keeping the common steady-state path short matters because the
    # CUDA events include the CPU gap between the event and kernel launch.
    # The specialized launcher performs the actual split-K/reducer work and
    # returns ``None`` for graph captures without a reusable workspace.
    if (
        os.environ.get("VLLM_HYQUANT_DECODE_INT8_QK", "1") != "0"
        and query.ndim == 3
        and kv_cache.ndim == 3
        and query.dtype == torch.bfloat16
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[1] > 0
        and query.shape[1] % kv_cache.shape[1] == 0
        and layout.block_size == 16
        and layout.anchor_count == 1
        and layout.group_size == 32
        and layout.num_groups == 4
        and layout.head_size == 128
        and query.shape[-1] == 128
        and query.shape[1] // kv_cache.shape[1] == 4
        and block_table.ndim == 2
        and block_table.shape[0] >= query.shape[0]
        and seq_lens.ndim == 1
        and seq_lens.shape[0] == query.shape[0]
        and prompt_lens.shape == seq_lens.shape
        and state_slots.shape == seq_lens.shape
    ):
        batch_size = int(query.shape[0])
        kvheads = int(kv_cache.shape[1])
        if max_seq_len is None:
            max_seq_len = max(1, int(seq_lens.max().item()))
        else:
            max_seq_len = max(1, int(max_seq_len))
        num_splits = get_hyquant_decode_split_count(batch_size, max_seq_len, 4)
        block_kv, block_h, num_warps = get_hyquant_decode_launch_config(
            batch_size, max_seq_len, 4
        )
        if (
            num_splits >= 1
            and block_h == 4
            and block_kv <= 128
            and (
                num_splits == 1
                or (
                    mid_o_buf is not None
                    and mid_o_buf.dtype == torch.float32
                    and mid_o_buf.device == query.device
                    and mid_o_buf.ndim == 4
                    and mid_o_buf.shape[0] >= batch_size
                    and mid_o_buf.shape[1] >= query.shape[1]
                    and mid_o_buf.shape[2] >= num_splits
                    and mid_o_buf.shape[3] >= 129
                )
            )
            and query.device == hot_key.device == hot_value.device
            and query.device == block_table.device == seq_lens.device
            and query.device == prompt_lens.device == state_slots.device
            and hot_key.shape == hot_value.shape
            and hot_key.ndim == 4
            and hot_key.shape[0] >= batch_size
            and hot_key.shape[1] == kvheads
            and hot_key.shape[3] == 128
            and hot_key.dtype == torch.bfloat16
            and (
                not fuse_current_store
                or (
                    current_key is not None
                    and current_value is not None
                    and current_key.shape == (batch_size, kvheads, 128)
                    and current_value.shape == (batch_size, kvheads, 128)
                )
            )
        ):
            from vllm.v1.attention.ops.hyquant_decode_int8 import (
                hyquant_int8qk_decode,
            )

            output_fast = torch.empty_like(query) if output is None else output
            if output_fast.shape == query.shape and output_fast.device == query.device:
                mid_fast = (
                    None
                    if num_splits == 1
                    else mid_o_buf[:batch_size, : query.shape[1], :num_splits, :129]
                )
                result = hyquant_int8qk_decode(
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
                    mid_fast,
                    num_splits,
                    block_kv,
                    block_h,
                    num_warps,
                    get_hyquant_decode_num_stages(batch_size, max_seq_len),
                    use_simt=False,
                    output=output_fast,
                    current_key=current_key,
                    current_value=current_value,
                    fuse_current_store=fuse_current_store,
                )
                _LAST_STATUS = "int8qk_groupdot_splitk"
                return result
    if query.ndim != 3 or seq_lens.ndim != 1 or query.shape[0] != seq_lens.shape[0]:
        _LAST_STATUS = "shape"
        return None
    if (
        kv_cache.ndim != 3
        or kv_cache.dtype != torch.uint8
        or kv_cache.shape[1] <= 0
        or kv_cache.shape[2] != layout.head_page_bytes
        or query.shape[1] % kv_cache.shape[1]
    ):
        _LAST_STATUS = "cache_shape"
        return None
    if (
        query.device != kv_cache.device
        or hot_key.device != query.device
        or hot_value.device != query.device
        or block_table.device != query.device
        or seq_lens.device != query.device
        or prompt_lens.device != query.device
        or state_slots.device != query.device
    ):
        _LAST_STATUS = "device"
        return None
    if (
        kv_cache.stride(-1) != 1
        or layout.head_page_bytes % 4
        or query.stride(-1) != 1
        or any(stride < 0 for stride in query.stride())
    ):
        _LAST_STATUS = "alignment_or_stride"
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
    if (
        hot_key.ndim != 4
        or hot_key.shape != hot_value.shape
        or hot_key.shape[0] < query.shape[0]
        or hot_key.shape[1] != kv_cache.shape[1]
        or hot_key.shape[3] != query.shape[-1]
        or hot_key.dtype != torch.bfloat16
        or hot_value.dtype != torch.bfloat16
        or hot_key.stride(-1) != 1
        or hot_value.stride(-1) != 1
    ):
        _LAST_STATUS = "hot_shape"
        return None
    if block_table.ndim != 2 or block_table.shape[0] < query.shape[0]:
        _LAST_STATUS = "block_table_shape"
        return None
    if prompt_lens.shape != seq_lens.shape or state_slots.shape != seq_lens.shape:
        _LAST_STATUS = "metadata_shape"
        return None

    # The direct decoder currently has four statically unrolled channel-group
    # accumulators.  Restricting the fast path to regular layouts is preferable
    # to generating an unexpectedly large register footprint or reading beyond
    # the packed last group for non-divisible head sizes.
    if (
        layout.block_size not in (16, 32, 64)
        or layout.head_size != query.shape[-1]
        or layout.head_size > 256
        or layout.anchor_count <= 0
        or layout.anchor_count >= layout.block_size
        or layout.quant_count <= 0
        or layout.group_size % 4
        or layout.head_size % layout.group_size
        or layout.num_groups > 4
    ):
        _LAST_STATUS = "unsupported_layout"
        return None

    batch_size = int(query.shape[0])
    kvheads = int(kv_cache.shape[1])
    query_group_size = int(query.shape[1] // kvheads)
    if max_seq_len is None:
        # ``seq_lens`` is a device tensor.  This scalar read occurs once per
        # attention call, while the result is reused by all kernel layers.
        max_seq_len = max(1, int(seq_lens.max().item()))
    else:
        max_seq_len = max(1, int(max_seq_len))
    num_splits = get_hyquant_decode_split_count(
        batch_size, max_seq_len, query_group_size
    )
    block_kv, block_h, num_warps = get_hyquant_decode_launch_config(
        batch_size, max_seq_len, query_group_size
    )
    if block_h > query_group_size:
        block_h = query_group_size
    if block_h not in (1, 2, 4, 8) or num_warps not in (1, 2, 4, 8):
        _LAST_STATUS = "launch_config"
        return None
    block_d = 1 << (query.shape[-1] - 1).bit_length()
    block_words = layout.group_size // 4
    if block_words <= 0 or block_words & (block_words - 1):
        _LAST_STATUS = "group_alignment"
        return None

    if fuse_current_store:
        expected_shape = (batch_size, kvheads, query.shape[-1])
        if (
            current_key is None
            or current_value is None
            or current_key.shape != expected_shape
            or current_value.shape != expected_shape
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
        # The arguments remain present in the Triton ABI; constexpr folding
        # removes all accesses when fusion is disabled.
        current_key = query
        current_value = query

    # A one-split launch writes the final output directly.  Only split-K
    # launches need the [D + LSE] partial workspace and reducer.
    if num_splits > 1:
        required_shape = (batch_size, query.shape[1], num_splits, block_d + 1)
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
            mid_o = mid_o_buf[:batch_size, : query.shape[1], :num_splits, : block_d + 1]
    else:
        # Generic one-split kernels also keep the partial pointer in their
        # ABI, although their constexpr direct-output branch does not touch
        # it.  Pass the final output as a valid placeholder for that fallback.
        mid_o = output
    cache_i16 = kv_cache.view(torch.int16)
    num_stages = int(get_hyquant_decode_num_stages(batch_size, max_seq_len))
    # The coalesced page-table path is valid when a tile spans no more than
    # the fixed metadata tile.  Keep the old load for unusually large tiles
    # (or an explicitly tuned BLOCK_KV=256) rather than risking an out of
    # bounds gather.
    coalesce_block_table = int(
        block_kv <= 128
        and os.environ.get("VLLM_HYQUANT_DECODE_COALESCE_BLOCK_TABLE", "1") != "0"
    )
    coalesce_scales = int(
        layout.num_groups == 4
        and os.environ.get("VLLM_HYQUANT_DECODE_COALESCE_SCALES", "1") != "0"
    )
    # The integer-QK specialization is tuned for the Qwen3-4B shape used by
    # the first production target.  It avoids the generic fused ABI entirely
    # and therefore keeps its register footprint low.  Other model/layout
    # combinations retain the BF16 group-dot implementation below.
    use_int8_specialized = (
        os.environ.get("VLLM_HYQUANT_DECODE_INT8_QK", "1") != "0"
        and layout.block_size == 16
        and layout.anchor_count == 1
        and layout.group_size == 32
        and layout.num_groups == 4
        and layout.head_size == 128
        and query_group_size == 4
        and block_h == 4
        and block_kv <= 128
        and num_splits >= 1
    )

    if use_int8_specialized:
        from vllm.v1.attention.ops.hyquant_decode_int8 import (
            hyquant_int8qk_decode,
        )

        result = hyquant_int8qk_decode(
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
            mid_o,
            num_splits,
            block_kv,
            block_h,
            num_warps,
            num_stages,
            use_simt=bool(os.environ.get("VLLM_HYQUANT_DECODE_SIMT", "0") == "1"),
            output=output,
            current_key=current_key,
            current_value=current_value,
            fuse_current_store=fuse_current_store,
        )
        _LAST_STATUS = "int8qk_groupdot_splitk"
        return result

    # Keep the generic kernel's optional integer-QK mode independent from the
    # fixed-shape specialization.  It is disabled by default because its
    # general ABI carries more state and is not consistently faster on other
    # head/group configurations.
    int8_qk = int(os.environ.get("VLLM_HYQUANT_DECODE_INT8_QK_GENERIC", "0") != "0")

    # The ordinary one-token decode path does not need the fused current-row
    # store.  Use a deliberately small kernel ABI in that case: keeping the
    # current pointers, direct-output branch, and generic scale branches out
    # of the Triton function materially reduces generated code and restores
    # the standalone group-dot performance.  The fused path below remains the
    # fallback for graph-safe in-kernel hot-ring publication.
    if (
        not fuse_current_store
        and num_splits > 1
        and layout.num_groups == 4
        and block_kv <= 128
        and block_h in (1, 2, 4, 8)
    ):
        from vllm.v1.attention.ops.hyquant_decode_fast import hyquant_fast_decode

        result = hyquant_fast_decode(
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
            mid_o,
            num_splits,
            block_kv,
            block_h,
            num_warps,
            num_stages,
            use_simt=bool(os.environ.get("VLLM_HYQUANT_DECODE_SIMT", "0") == "1"),
            output=output,
        )
        _LAST_STATUS = "fast_groupdot_splitk"
        return result

    group_tiles = (query_group_size + block_h - 1) // block_h
    _hyquant_groupdot_split_decode_kernel[
        (batch_size, kvheads, group_tiles * num_splits)
    ](
        query,
        current_key,
        current_value,
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
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        num_kv_heads=kvheads,
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
        block_d=block_d,
        block_kv=block_kv,
        block_words=block_words,
        # A 128-row tile crosses at most nine 15-row compact pages.  Keeping
        # this tile at 16 entries is enough for the tuned path and avoids
        # inflating register usage; the 256-row tuning path disables the
        # coalesced gather above.
        block_table_tile=16,
        coalesce_block_table=coalesce_block_table,
        coalesce_scales=coalesce_scales,
        num_splits=num_splits,
        use_simt=int(os.environ.get("VLLM_HYQUANT_DECODE_SIMT", "0") == "1"),
        int8_qk=int8_qk,
        fuse_current_store=int(fuse_current_store),
        softmax_scale2=scale * _LOG2_E,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if num_splits >= 1:
        _hyquant_grouped_split_reduce_kernel[(batch_size, query.shape[1])](
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
        _LAST_STATUS = "groupdot_splitk"
    else:
        _LAST_STATUS = "groupdot"
    return output


# Keep the old descriptive name as a local compatibility facade for scripts
# that imported the experimental module before it became production code.
triton_hyquant_grouped_decode_attention = triton_hyquant_groupdot_decode_attention
