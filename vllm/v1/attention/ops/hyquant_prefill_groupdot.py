"""Grouped direct-page prefill kernel for the production HyQuant layout."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.ops.hyquant_common import HyQuantBlockLayout

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453


if HAS_TRITON:

    @triton.jit(do_not_specialize=["stride_block_table"])
    def _hyquant_prefix_groupdot_kernel(
        query_ptr,
        cache_i16_ptr,
        block_table_ptr,
        query_start_loc_ptr,
        prefix_lens_ptr,
        output_ptr,
        lse_ptr,
        stride_query_token,
        stride_query_head,
        stride_cache_block,
        stride_cache_head,
        stride_block_table,
        stride_output_token,
        stride_output_head,
        stride_lse_head,
        stride_lse_token,
        query_group_size: tl.constexpr,
        head_size: tl.constexpr,
        block_size: tl.constexpr,
        anchor_count: tl.constexpr,
        group_size: tl.constexpr,
        num_groups: tl.constexpr,
        packed_value_bytes: tl.constexpr,
        anchor_key_offset: tl.constexpr,
        anchor_value_offset: tl.constexpr,
        quant_key_offset: tl.constexpr,
        quant_value_offset: tl.constexpr,
        key_scale_offset: tl.constexpr,
        value_scale_offset: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
        block_rows: tl.constexpr,
        block_kv: tl.constexpr,
        block_words: tl.constexpr,
        block_table_tile: tl.constexpr,
        softmax_scale2: tl.constexpr,
        ln2: tl.constexpr,
    ):
        request_index = tl.program_id(0)
        query_tile = tl.program_id(1)
        head_tile = tl.program_id(2)
        head_tiles_per_kv = tl.cdiv(query_group_size, block_h)
        kv_head = head_tile // head_tiles_per_kv
        group_head_start = (head_tile % head_tiles_per_kv) * block_h

        query_start = tl.load(query_start_loc_ptr + request_index).to(tl.int64)
        query_end = tl.load(query_start_loc_ptr + request_index + 1).to(tl.int64)
        prefix_length = tl.load(prefix_lens_ptr + request_index).to(tl.int64)
        prefix_blocks = prefix_length // block_size
        table_base = request_index * stride_block_table

        row_offsets = tl.arange(0, block_rows)
        query_lanes = row_offsets // block_h
        head_lanes = row_offsets % block_h
        query_tokens = query_start + query_tile * block_m + query_lanes
        group_heads = group_head_start + head_lanes
        query_heads = kv_head * query_group_size + group_heads
        valid_query = (
            (query_lanes < block_m)
            & (query_tokens < query_end)
            & (group_heads < query_group_size)
        )
        d_offsets = tl.arange(0, group_size)
        query_bases = (
            query_tokens[:, None] * stride_query_token
            + query_heads[:, None] * stride_query_head
        )
        q0 = tl.load(
            query_ptr + query_bases + d_offsets[None, :],
            mask=valid_query[:, None],
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)
        q1 = tl.load(
            query_ptr + query_bases + group_size + d_offsets[None, :],
            mask=valid_query[:, None],
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)
        q2 = tl.load(
            query_ptr + query_bases + 2 * group_size + d_offsets[None, :],
            mask=valid_query[:, None],
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)
        q3 = tl.load(
            query_ptr + query_bases + 3 * group_size + d_offsets[None, :],
            mask=valid_query[:, None],
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.bfloat16)

        max_score = tl.full([block_rows], -float("inf"), tl.float32)
        normalizer = tl.zeros([block_rows], tl.float32)
        acc0 = tl.zeros([block_rows, group_size], tl.float32)
        acc1 = tl.zeros([block_rows, group_size], tl.float32)
        acc2 = tl.zeros([block_rows, group_size], tl.float32)
        acc3 = tl.zeros([block_rows, group_size], tl.float32)
        kv_offsets = tl.arange(0, block_kv)
        word_offsets = tl.arange(0, block_words)

        quant_per_block = block_size - anchor_count
        quant_count = prefix_blocks * quant_per_block
        for start in tl.range(0, quant_count, block_kv):
            quant_indices = start + kv_offsets
            valid_kv = quant_indices < quant_count
            logical_blocks = quant_indices // quant_per_block
            quant_rows = quant_indices % quant_per_block
            first_block = start // quant_per_block
            table_offsets = tl.arange(0, block_table_tile)
            table_values = tl.load(
                block_table_ptr + table_base + first_block + table_offsets,
                mask=first_block + table_offsets < prefix_blocks,
                other=0,
            ).to(tl.int64)
            table_select = tl.minimum(
                tl.maximum(logical_blocks - first_block, 0), block_table_tile - 1
            )
            physical_blocks = tl.gather(table_values, table_select, axis=0)
            page_bases = (
                physical_blocks * stride_cache_block + kv_head * stride_cache_head
            )

            scores = tl.zeros([block_rows, block_kv], tl.float32)
            for group_index in range(num_groups):
                packed = (
                    tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + quant_key_offset) // 2
                        + quant_rows[:, None] * (packed_value_bytes // 2)
                        + group_index * block_words
                        + word_offsets[None, :],
                        mask=valid_kv[:, None],
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
                if group_index == 0:
                    query_group = q0
                elif group_index == 1:
                    query_group = q1
                elif group_index == 2:
                    query_group = q2
                else:
                    query_group = q3
                key_scale_bits = tl.load(
                    cache_i16_ptr
                    + (page_bases + key_scale_offset) // 2
                    + quant_rows * num_groups
                    + group_index,
                    mask=valid_kv,
                    other=0,
                    cache_modifier=".cg",
                )
                key_scale = key_scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
                scores += (
                    tl.dot(query_group, tl.trans(codes)).to(tl.float32)
                    * key_scale[None, :]
                )

            score_mask = valid_query[:, None] & valid_kv[None, :]
            scores = tl.where(score_mask, scores * softmax_scale2, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_score, tile_max)
            old_factor = tl.where(
                valid_query & (new_max > -float("inf")),
                tl.math.exp2(max_score - new_max),
                0.0,
            )
            probabilities = tl.math.exp2(scores - new_max[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            normalizer = normalizer * old_factor + tl.sum(probabilities, axis=1)

            for group_index in range(num_groups):
                packed = (
                    tl.load(
                        cache_i16_ptr
                        + (page_bases[:, None] + quant_value_offset) // 2
                        + quant_rows[:, None] * (packed_value_bytes // 2)
                        + group_index * block_words
                        + word_offsets[None, :],
                        mask=valid_kv[:, None],
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
                value_scale_bits = tl.load(
                    cache_i16_ptr
                    + (page_bases + value_scale_offset) // 2
                    + quant_rows * num_groups
                    + group_index,
                    mask=valid_kv,
                    other=0,
                    cache_modifier=".cg",
                )
                value_scale = value_scale_bits.to(tl.float16, bitcast=True).to(
                    tl.float32
                )
                update = tl.dot(
                    (probabilities * value_scale[None, :]).to(tl.bfloat16), codes
                ).to(tl.float32)
                if group_index == 0:
                    acc0 = acc0 * old_factor[:, None] + update
                elif group_index == 1:
                    acc1 = acc1 * old_factor[:, None] + update
                elif group_index == 2:
                    acc2 = acc2 * old_factor[:, None] + update
                else:
                    acc3 = acc3 * old_factor[:, None] + update
            max_score = new_max

        anchor_count_total = prefix_blocks * anchor_count
        for start in tl.range(0, anchor_count_total, block_kv):
            anchor_indices = start + kv_offsets
            valid_kv = anchor_indices < anchor_count_total
            logical_blocks = anchor_indices // anchor_count
            anchor_rows = anchor_indices % anchor_count
            physical_blocks = tl.load(
                block_table_ptr + table_base + logical_blocks,
                mask=valid_kv,
                other=0,
            ).to(tl.int64)
            page_bases = (
                physical_blocks * stride_cache_block + kv_head * stride_cache_head
            )
            key_bases = (page_bases + anchor_key_offset) // 2 + anchor_rows * head_size
            scores = tl.zeros([block_rows, block_kv], tl.float32)
            for group_index in range(num_groups):
                keys = tl.load(
                    cache_i16_ptr
                    + key_bases[:, None]
                    + group_index * group_size
                    + d_offsets[None, :],
                    mask=valid_kv[:, None],
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.bfloat16, bitcast=True)
                if group_index == 0:
                    query_group = q0
                elif group_index == 1:
                    query_group = q1
                elif group_index == 2:
                    query_group = q2
                else:
                    query_group = q3
                scores += tl.dot(query_group, tl.trans(keys)).to(tl.float32)
            score_mask = valid_query[:, None] & valid_kv[None, :]
            scores = tl.where(score_mask, scores * softmax_scale2, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_score, tile_max)
            old_factor = tl.where(
                valid_query & (new_max > -float("inf")),
                tl.math.exp2(max_score - new_max),
                0.0,
            )
            probabilities = tl.math.exp2(scores - new_max[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            normalizer = normalizer * old_factor + tl.sum(probabilities, axis=1)
            value_bases = (
                page_bases + anchor_value_offset
            ) // 2 + anchor_rows * head_size
            for group_index in range(num_groups):
                values = tl.load(
                    cache_i16_ptr
                    + value_bases[:, None]
                    + group_index * group_size
                    + d_offsets[None, :],
                    mask=valid_kv[:, None],
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.bfloat16, bitcast=True)
                update = tl.dot(probabilities.to(tl.bfloat16), values).to(tl.float32)
                if group_index == 0:
                    acc0 = acc0 * old_factor[:, None] + update
                elif group_index == 1:
                    acc1 = acc1 * old_factor[:, None] + update
                elif group_index == 2:
                    acc2 = acc2 * old_factor[:, None] + update
                else:
                    acc3 = acc3 * old_factor[:, None] + update
            max_score = new_max

        safe_normalizer = tl.maximum(normalizer, 1e-20)
        output_bases = (
            query_tokens[:, None] * stride_output_token
            + query_heads[:, None] * stride_output_head
        )
        tl.store(
            output_ptr + output_bases + d_offsets[None, :],
            acc0 / safe_normalizer[:, None],
            mask=valid_query[:, None],
        )
        tl.store(
            output_ptr + output_bases + group_size + d_offsets[None, :],
            acc1 / safe_normalizer[:, None],
            mask=valid_query[:, None],
        )
        tl.store(
            output_ptr + output_bases + 2 * group_size + d_offsets[None, :],
            acc2 / safe_normalizer[:, None],
            mask=valid_query[:, None],
        )
        tl.store(
            output_ptr + output_bases + 3 * group_size + d_offsets[None, :],
            acc3 / safe_normalizer[:, None],
            mask=valid_query[:, None],
        )
        lse = tl.where(
            normalizer > 0.0,
            (max_score + tl.math.log2(safe_normalizer)) * ln2,
            -float("inf"),
        )
        tl.store(
            lse_ptr + query_heads * stride_lse_head + query_tokens * stride_lse_token,
            lse,
            mask=valid_query,
        )


def get_groupdot_launch_config(query_group_size: int) -> tuple[int, ...]:
    block_h = min(query_group_size, 4)
    block_m = 16 if block_h == 1 else 8
    block_rows = triton.next_power_of_2(max(16, block_m * block_h))
    block_kv = 128
    block_table_tile = 16
    num_warps = 4
    return block_m, block_h, block_rows, block_kv, block_table_tile, num_warps


def triton_hyquant_prefix_groupdot(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefix_lens: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    layout: HyQuantBlockLayout,
    scale: float,
    max_query_len: int,
) -> bool:
    """Fill a prefix partial for the common 1-anchor, group-32 layout."""
    if (
        not HAS_TRITON
        or layout.block_size != 16
        or layout.head_size != 128
        or layout.group_size != 32
        or layout.num_groups != 4
        or layout.anchor_count != 1
        or layout.quant_count != 15
    ):
        return False
    query_group_size = query.shape[1] // kv_cache.shape[1]
    block_m, block_h, block_rows, block_kv, table_tile, num_warps = (
        get_groupdot_launch_config(query_group_size)
    )
    grid = (
        prefix_lens.shape[0],
        triton.cdiv(max_query_len, block_m),
        kv_cache.shape[1] * triton.cdiv(query_group_size, block_h),
    )
    cache_i16 = kv_cache.view(torch.int16)
    _hyquant_prefix_groupdot_kernel[grid](
        query,
        cache_i16,
        block_table,
        query_start_loc,
        prefix_lens,
        output,
        lse,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        lse.stride(1),
        query_group_size=query_group_size,
        head_size=layout.head_size,
        block_size=layout.block_size,
        anchor_count=layout.anchor_count,
        group_size=layout.group_size,
        num_groups=layout.num_groups,
        packed_value_bytes=layout.packed_value_bytes,
        anchor_key_offset=layout.anchor_key_offset_bytes,
        anchor_value_offset=layout.anchor_value_offset_bytes,
        quant_key_offset=layout.quant_key_offset_bytes,
        quant_value_offset=layout.quant_value_offset_bytes,
        key_scale_offset=layout.key_scale_offset_bytes,
        value_scale_offset=layout.value_scale_offset_bytes,
        block_m=block_m,
        block_h=block_h,
        block_rows=block_rows,
        block_kv=block_kv,
        block_words=layout.group_size // 4,
        block_table_tile=table_tile,
        softmax_scale2=scale * _LOG2E,
        ln2=_LN2,
        num_warps=num_warps,
        num_stages=2,
    )
    return True
