"""Low-register split-K decode specialization for HyQuant.

This kernel intentionally has no current-token store or direct-output branch.
Keeping the hot path small lets Triton generate the same compact code shape
that was validated in the standalone A/B benchmark.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.ops.hyquant_common import HyQuantBlockLayout

if HAS_TRITON:

    @triton.jit
    def _groupdot_kernel(
        query_ptr,
        cache_i16_ptr,
        hot_key_ptr,
        hot_value_ptr,
        block_table_ptr,
        seq_lens_ptr,
        prompt_lens_ptr,
        state_slots_ptr,
        mid_o_ptr,
        stride_query_token,
        stride_query_head,
        stride_cache_block,
        stride_cache_head,
        stride_hot_slot,
        stride_hot_head,
        stride_hot_token,
        stride_block_table,
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
        block_d: tl.constexpr,
        block_kv: tl.constexpr,
        block_words: tl.constexpr,
        block_table_tile: tl.constexpr,
        coalesce_block_table: tl.constexpr,
        num_splits: tl.constexpr,
        use_simt: tl.constexpr,
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
            mask=h_mask[:, None] & d_mask[None, :] & ((d_offsets)[None, :] < head_size),
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

        quant_per_block = block_size - anchor_count
        quant_count = committed_blocks * quant_per_block
        quant_tiles = tl.cdiv(quant_count, block_kv)
        tiles_per_split = tl.cdiv(quant_tiles, num_splits)
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
            page_bases = physical_blocks * stride_cache_block + (
                kv_head * stride_cache_head
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
                if g == 0:
                    key_scale = key_scale_0
                elif g == 1:
                    key_scale = key_scale_1
                elif g == 2:
                    key_scale = key_scale_2
                else:
                    key_scale = key_scale_3
                scores += raw_score * key_scale[None, :]

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

            # PV: reuse the same integer representation.  Scaling P by the
            # per-row V scale before the MMA avoids constructing BF16 values.
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
                if g == 0:
                    value_scale = value_scale_0
                elif g == 1:
                    value_scale = value_scale_1
                elif g == 2:
                    value_scale = value_scale_2
                else:
                    value_scale = value_scale_3
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

        # Recent BF16 rows live in the ring.  This path is normally only a
        # handful of tiles and is intentionally kept straightforward.
        hot_count = sequence_length - committed
        hot_tiles = tl.cdiv(hot_count, block_kv)
        hot_tiles_per_split = tl.cdiv(hot_tiles, num_splits)
        hot_start = split_index * hot_tiles_per_split * block_kv
        hot_end = tl.minimum(hot_start + hot_tiles_per_split * block_kv, hot_count)
        for start in tl.range(hot_start, hot_end, block_kv):
            indices = start + kv_offsets
            valid = indices < hot_count
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

        # Store the group accumulators into the common [D + LSE] workspace.
        result_bases = (
            request_index * stride_mid_batch
            + query_heads[:, None] * stride_mid_head
            + split_index * stride_mid_split
        )
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
            tl.store(
                mid_o_ptr + result_bases + g * group_size + d_offsets[None, :],
                acc_out * inv_norm,
                mask=h_mask[:, None]
                & d_mask[None, :]
                & ((g * group_size + d_offsets)[None, :] < head_size),
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
            mask=h_mask,
        )


def hyquant_fast_decode(
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
    workspace: torch.Tensor,
    num_splits: int,
    block_kv: int,
    block_h: int,
    num_warps: int,
    num_stages: int,
    use_simt: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the compact split-K kernel and the shared reducer."""
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
    mid = workspace[:bsz, :qheads, :num_splits, : block_d + 1]
    cache_i16 = kv_cache.view(torch.int16)
    _groupdot_kernel[(bsz, kvheads, ((qgroup + block_h - 1) // block_h) * num_splits)](
        query,
        cache_i16,
        hot_key,
        hot_value,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        mid,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        hot_key.stride(0),
        hot_key.stride(1),
        hot_key.stride(2),
        block_table.stride(0),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
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
        softmax_scale2=scale * 1.4426950408889634,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    from vllm.v1.attention.ops.hyquant_decode_groupdot import (
        _hyquant_grouped_split_reduce_kernel,
    )

    out = torch.empty_like(query) if output is None else output
    if out.shape != query.shape or out.device != query.device:
        raise ValueError("output must have the same shape and device as query")
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
