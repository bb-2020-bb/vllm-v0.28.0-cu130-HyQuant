"""Low-register split-K decode specialization for HyQuant.

This kernel intentionally has no current-token store or direct-output branch.
Keeping the hot path small lets Triton generate the same compact code shape
that was validated in the standalone A/B benchmark.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.ops.hyquant_common import HyQuantBlockLayout

_WARMED_INT8QK_KERNELS: set[tuple] = set()

if HAS_TRITON:

    @triton.jit
    def _int8qk_kernel(
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
        mid_o_ptr,
        output_ptr,
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
        num_splits: tl.constexpr,
        use_simt: tl.constexpr,
        fuse_current_store: tl.constexpr,
        softmax_scale2: tl.constexpr,
    ):
        request_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        group_split = tl.program_id(2)
        split_index = group_split % num_splits
        group_index = group_split // num_splits

        sequence_length = tl.load(seq_lens_ptr + request_index).to(tl.int64)
        active_splits = num_splits

        h_offsets = tl.arange(0, block_h)
        group_start = group_index * block_h
        query_heads = kv_head * query_group_size + group_start + h_offsets
        h_mask = h_offsets < (query_group_size - group_start)
        d_offsets = tl.arange(0, group_size)
        d_mask = d_offsets < group_size
        kv_offsets = tl.arange(0, block_kv)
        word_offsets = tl.arange(0, block_words)
        scale_offsets = tl.arange(0, 4)

        # Keep the query in four small group tiles.  This is the key
        # difference from the production kernel, which materialises a
        # [BLOCK_KV, D] BF16 K/V tile before each dot.
        query_bases = (
            request_index * stride_query_token
            + query_heads[:, None] * stride_query_head
        )
        q0 = tl.load(
            query_ptr + query_bases + d_offsets[None, :],
            mask=h_mask[:, None] & d_mask[None, :] & (d_offsets[None, :] < head_size),
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)
        q1 = tl.load(
            query_ptr + query_bases + group_size + d_offsets[None, :],
            mask=h_mask[:, None]
            & d_mask[None, :]
            & ((group_size + d_offsets)[None, :] < head_size),
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)
        q2 = tl.load(
            query_ptr + query_bases + 2 * group_size + d_offsets[None, :],
            mask=h_mask[:, None]
            & d_mask[None, :]
            & ((2 * group_size + d_offsets)[None, :] < head_size),
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)
        q3 = tl.load(
            query_ptr + query_bases + 3 * group_size + d_offsets[None, :],
            mask=h_mask[:, None]
            & d_mask[None, :]
            & ((3 * group_size + d_offsets)[None, :] < head_size),
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)
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

        quant_per_block = block_size - anchor_count
        quant_count = committed_blocks * quant_per_block
        quant_tiles = tl.cdiv(quant_count, block_kv)
        tiles_per_split = tl.cdiv(quant_tiles, active_splits)
        quant_start = split_index * tiles_per_split * block_kv
        quant_end = tl.minimum(quant_start + tiles_per_split * block_kv, quant_count)

        for start in tl.range(quant_start, quant_end, block_kv):
            quant_indices = start + kv_offsets
            valid = quant_indices < quant_count
            logical_blocks = quant_indices // quant_per_block
            quant_rows = quant_indices % quant_per_block
            if coalesce_block_table:
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

            # Load the contiguous per-token group scales as two coalesced
            # matrices.  The old path issued one strided vector load for every
            # group, which generated eight independent memory instructions per
            # tile for K/V.
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
            value_scales_all = value_scale_bits_all.to(tl.float16, bitcast=True).to(
                tl.float32
            )
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

            # QK: four independent group MMA operations over raw signed INT4
            # codes.  The scale is one scalar per token/group, so it is applied
            # to the resulting score instead of being expanded to D channels.
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
                if g == 0:
                    key_scale = key_scale_0
                elif g == 1:
                    key_scale = key_scale_1
                elif g == 2:
                    key_scale = key_scale_2
                else:
                    key_scale = key_scale_3
                scores += raw_score * (qscale[:, None] * key_scale[None, :])

            score_mask = h_mask[:, None] & valid[None, :]
            scores = scores * softmax_scale2
            scores = tl.where(score_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_score, tile_max)
            old_factor = tl.where(
                new_max > -float("inf"), tl.math.exp2(max_score - new_max), 0.0
            )
            probabilities = tl.math.exp2(scores - new_max[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            normalizer = normalizer * old_factor + tl.sum(probabilities, axis=1)

            # PV: use the packed integer representation, converting only the
            # small tile to BF16 for the probability/value MMA.
            for g in range(num_groups):
                if g == 0:
                    value_scale = value_scale_0
                elif g == 1:
                    value_scale = value_scale_1
                elif g == 2:
                    value_scale = value_scale_2
                else:
                    value_scale = value_scale_3
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
                codes_i8 = tl.reshape(
                    tl.join(tl.join(n0, n2), tl.join(n1, n3)),
                    (block_kv, group_size),
                ).to(tl.int8)
                codes = codes_i8.to(tl.bfloat16)
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
        anchor_tiles_per_split = tl.cdiv(anchor_tiles, active_splits)
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

        # Recent BF16 rows live in the ring.  This path is normally only a
        # handful of tiles and is intentionally kept straightforward.
        hot_count = sequence_length - committed
        hot_scan_count = (
            tl.maximum(hot_count - 1, 0) if fuse_current_store else hot_count
        )
        hot_tiles = tl.cdiv(hot_scan_count, block_kv)
        hot_tiles_per_split = tl.cdiv(hot_tiles, active_splits)
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
            # The split that owns the final hot tile also consumes and stores
            # the current row.  This preserves exact split-K softmax merging
            # while removing the separate hot-store launch.
            current_owner = tl.minimum(
                tl.maximum(hot_count - 1, 0)
                // tl.maximum(hot_tiles_per_split * block_kv, 1),
                active_splits - 1,
            )
            current_active = (state_slot >= 0) & (split_index == current_owner)
            current_key_base = (
                request_index * stride_current_key_token
                + kv_head * stride_current_key_head
            )
            current_value_base = (
                request_index * stride_current_value_token
                + kv_head * stride_current_value_head
            )
            current_ring_base = (
                state_slot * stride_hot_slot
                + kv_head * stride_hot_head
                + ((sequence_length - 1) % hot_capacity) * stride_hot_token
            )
            current_scores = tl.zeros([block_h], tl.float32)
            for g in range(num_groups):
                if g == 0:
                    qg = q0
                elif g == 1:
                    qg = q1
                elif g == 2:
                    qg = q2
                else:
                    qg = q3
                ck = tl.load(
                    current_key_ptr + current_key_base + g * group_size + d_offsets,
                    mask=current_active & d_mask,
                    other=0.0,
                ).to(tl.bfloat16)
                current_scores += tl.sum(
                    qg.to(tl.float32) * ck[None, :].to(tl.float32), axis=1
                )
                tl.store(
                    hot_key_ptr + current_ring_base + g * group_size + d_offsets,
                    ck,
                    mask=current_active & d_mask,
                )
            current_scores *= softmax_scale2
            current_mask = h_mask & current_active
            current_scores = tl.where(current_mask, current_scores, -float("inf"))
            new_max = tl.maximum(max_score, current_scores)
            old_factor = tl.where(
                new_max > -float("inf"), tl.math.exp2(max_score - new_max), 0.0
            )
            current_probabilities = tl.math.exp2(current_scores - new_max)
            current_probabilities = tl.where(current_mask, current_probabilities, 0.0)
            normalizer = normalizer * old_factor + current_probabilities
            for g in range(num_groups):
                cv = tl.load(
                    current_value_ptr + current_value_base + g * group_size + d_offsets,
                    mask=current_active & d_mask,
                    other=0.0,
                ).to(tl.bfloat16)
                tl.store(
                    hot_value_ptr + current_ring_base + g * group_size + d_offsets,
                    cv,
                    mask=current_active & d_mask,
                )
                current_update = current_probabilities[:, None] * cv[None, :].to(
                    tl.float32
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

        # A single split has no merge work.  Write the normalized result
        # directly to the caller's output; this removes a workspace write,
        # a reducer read, and one kernel launch from short-context decode.
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
            else:
                result_bases = (
                    request_index * stride_mid_batch
                    + query_heads[:, None] * stride_mid_head
                    + split_index * stride_mid_split
                    + g * group_size
                )
            if num_splits == 1:
                tl.store(
                    output_ptr + result_bases + d_offsets[None, :],
                    acc_out * inv_norm,
                    mask=h_mask[:, None]
                    & d_mask[None, :]
                    & ((g * group_size + d_offsets)[None, :] < head_size),
                )
            else:
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


def hyquant_int8qk_decode(
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
    workspace: torch.Tensor | None,
    num_splits: int,
    block_kv: int,
    block_h: int,
    num_warps: int,
    num_stages: int,
    use_simt: bool = False,
    output: torch.Tensor | None = None,
    current_key: torch.Tensor | None = None,
    current_value: torch.Tensor | None = None,
    fuse_current_store: bool = False,
) -> torch.Tensor:
    """Run the compact split-K kernel, merging only when splits require it."""
    if not HAS_TRITON:
        raise RuntimeError("Triton unavailable")
    bsz, qheads, dim = query.shape
    kvheads = kv_cache.shape[1]
    qgroup = qheads // kvheads
    block_d = 1 << (dim - 1).bit_length()
    block_words = layout.group_size // 4
    if block_h not in (1, 2, 4, 8):
        raise ValueError("block_h must be one of 1, 2, 4, 8")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, 8")
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")
    out = torch.empty_like(query) if output is None else output
    if out.shape != query.shape or out.device != query.device:
        raise ValueError("output must have the same shape and device as query")
    # The one-split kernel never dereferences the partial pointer.  Reusing the
    # final output as its ABI placeholder lets callers omit a workspace while
    # keeping the compiled signature uniform.
    if num_splits == 1:
        mid = out
    else:
        if workspace is None:
            raise ValueError("workspace is required when num_splits > 1")
        mid = workspace[:bsz, :qheads, :num_splits, : block_d + 1]
    cache_i16 = kv_cache.view(torch.int16)
    if current_key is None:
        current_key = query
    if current_value is None:
        current_value = query
    if fuse_current_store:
        expected = (bsz, kvheads, dim)
        if current_key.shape != expected or current_value.shape != expected:
            raise ValueError("invalid current K/V shape for fused decode")
    _int8qk_kernel[(bsz, kvheads, ((qgroup + block_h - 1) // block_h) * num_splits)](
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
        mid,
        out,
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
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        out.stride(0),
        out.stride(1),
        num_kv_heads=kvheads,
        query_group_size=qgroup,
        head_size=dim,
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
        block_table_tile=16,
        coalesce_block_table=True,
        num_splits=num_splits,
        use_simt=int(use_simt),
        fuse_current_store=int(fuse_current_store),
        softmax_scale2=scale * 1.4426950408889634,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if num_splits > 1:
        from vllm.v1.attention.ops.hyquant_decode_groupdot import (
            _hyquant_grouped_split_reduce_kernel,
        )

        _hyquant_grouped_split_reduce_kernel[(bsz, qheads)](
            mid,
            out,
            mid.stride(0),
            mid.stride(1),
            mid.stride(2),
            out.stride(0),
            out.stride(1),
            head_size=dim,
            num_splits=num_splits,
            block_d=block_d,
            num_warps=4,
            num_stages=1,
        )
    return out


def get_hyquant_int8qk_warmup_configs(
    max_seq_len: int,
    max_batch_size: int,
    query_group_size: int = 4,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return one representative shape for every reachable decode config.

    Batch size and sequence length select static split/tile configurations,
    but neither value belongs in the kernel specialization itself.  Enumerate
    the policy boundaries so model setup compiles each reachable configuration
    once without allocating a synthetic maximum-size batch.
    """
    if max_seq_len <= 0 or max_batch_size <= 0:
        return ()

    from vllm.v1.attention.ops.hyquant_decode_groupdot import (
        get_hyquant_decode_launch_config,
        get_hyquant_decode_num_stages,
        get_hyquant_decode_split_count,
    )

    batch_points = [1]
    batch_points.extend(
        boundary for boundary in (2, 3, 9, 17) if boundary <= max_batch_size
    )
    sequence_points = sorted(
        {
            1,
            max_seq_len,
            *(
                boundary
                for boundary in (512, 2048, 4096, 6144, 8192, 16384)
                if boundary <= max_seq_len
            ),
        }
    )
    representative_by_config: dict[tuple[int, int, int, int, int], int] = {}
    for batch_size in batch_points:
        for sequence_length in sequence_points:
            num_splits = get_hyquant_decode_split_count(
                batch_size, sequence_length, query_group_size
            )
            block_kv, block_h, num_warps = get_hyquant_decode_launch_config(
                batch_size, sequence_length, query_group_size
            )
            if block_h != 4 or block_kv > 128:
                continue
            config = (
                num_splits,
                block_kv,
                block_h,
                num_warps,
                get_hyquant_decode_num_stages(batch_size, sequence_length),
            )
            representative_by_config.setdefault(config, sequence_length)
    return tuple(
        (sequence_length, *config)
        for config, sequence_length in representative_by_config.items()
    )


def warmup_hyquant_int8qk_kernel(
    layout: HyQuantBlockLayout,
    num_kv_heads: int,
    num_query_heads: int,
    window_size: int,
    retire_interval: int,
    hot_capacity: int,
    scale: float,
    device: torch.device,
    max_seq_len: int = 32768,
    max_state_slots: int = 1,
    block_table_width: int | None = None,
    fuse_current_store: bool = True,
) -> None:
    """Compile all reachable fixed-shape INT8-QK variants during setup.

    Static split counts preserve the validated steady-state performance, so
    each policy bucket needs its own specialization.  The synthetic launch is
    always batch one because batch size only changes the grid; the configured
    maximum batch is used solely to discover reachable launch configurations.
    """
    if not HAS_TRITON or device.type != "cuda":
        return
    configs = get_hyquant_int8qk_warmup_configs(
        max_seq_len, max_state_slots, num_query_heads // num_kv_heads
    )
    key = (
        device.index,
        layout,
        num_kv_heads,
        num_query_heads,
        window_size,
        retire_interval,
        hot_capacity,
        block_table_width,
        configs,
        fuse_current_store,
    )
    if key in _WARMED_INT8QK_KERNELS:
        return
    if (
        layout.block_size != 16
        or layout.head_size != 128
        or layout.group_size != 32
        or layout.num_groups != 4
        or layout.anchor_count != 1
        or num_query_heads // num_kv_heads != 4
    ):
        return
    try:
        table_width = block_table_width or (
            (max_seq_len + layout.block_size - 1) // layout.block_size
        )
        table_width = max(1, table_width)
        cache = torch.zeros(
            table_width,
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
        current_key = torch.zeros(
            1, num_kv_heads, layout.head_size, dtype=torch.bfloat16, device=device
        )
        current_value = torch.zeros_like(current_key)
        block_table = torch.arange(table_width, dtype=torch.int32, device=device)[None]
        seq_lens = torch.ones(1, dtype=torch.int32, device=device)
        prompt_lens = torch.ones_like(seq_lens)
        state_slots = torch.zeros(1, dtype=torch.int32, device=device)
        workspace = torch.empty(
            1,
            num_query_heads,
            max((config[1] for config in configs), default=1),
            129,
            dtype=torch.float32,
            device=device,
        )
        output = torch.empty_like(query)
        for (
            sequence_length,
            num_splits,
            block_kv,
            block_h,
            num_warps,
            num_stages,
        ) in configs:
            seq_lens.fill_(sequence_length)
            prompt_lens.fill_(max(1, sequence_length - 1))
            hyquant_int8qk_decode(
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
                workspace,
                num_splits,
                block_kv,
                block_h,
                num_warps,
                num_stages,
                output=output,
                current_key=current_key,
                current_value=current_value,
                fuse_current_store=fuse_current_store,
            )
            torch.cuda.synchronize(device)
        _WARMED_INT8QK_KERNELS.add(key)
    except Exception:
        # Warmup is advisory; deployment can still compile lazily on a target
        # with a different Triton ABI or a locked workspace configuration.
        return
