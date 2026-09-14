"""Direct compact-page attention for HyQuant prefix-cache hits.

The cached prefix stays in its compact K4/V4 page format. This module computes
a partial attention result and natural-log LSE directly from those pages. The
caller merges it with the BF16 suffix result produced by FlashAttention.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.ops.hyquant_common import (
    HYQUANT_QUANT_FLAG,
    HyQuantBlockLayout,
)
from vllm.v1.attention.ops.hyquant_prefill_groupdot import (
    triton_hyquant_prefix_groupdot,
)

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_LAST_STATUS = "not_run"
_WARMED_KERNELS: set[tuple] = set()


def get_hyquant_prefix_status() -> str:
    """Return the status of the most recent direct-prefix launch."""
    return _LAST_STATUS


if HAS_TRITON:

    @triton.jit(
        do_not_specialize=[
            "stride_query_token",
            "stride_query_head",
            "stride_cache_block",
            "stride_cache_head",
            "stride_cache_i16_block",
            "stride_cache_i16_head",
            "stride_block_table",
            "stride_output_token",
            "stride_output_head",
            "stride_lse_head",
            "stride_lse_token",
        ],
        do_not_specialize_on_alignment=[
            "query_ptr",
            "cache_ptr",
            "cache_i16_ptr",
            "block_table_ptr",
            "query_start_loc_ptr",
            "prefix_lens_ptr",
            "output_ptr",
            "lse_ptr",
        ],
    )
    def _hyquant_prefix_tiled_kernel(
        query_ptr,
        cache_ptr,
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
        stride_cache_i16_block,
        stride_cache_i16_head,
        stride_block_table,
        stride_output_token,
        stride_output_head,
        stride_lse_head,
        stride_lse_token,
        query_group_size: tl.constexpr,
        head_size: tl.constexpr,
        block_size: tl.constexpr,
        anchor_count: tl.constexpr,
        quant_count: tl.constexpr,
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
        block_m: tl.constexpr,
        block_h: tl.constexpr,
        block_rows: tl.constexpr,
        block_kv: tl.constexpr,
        block_d: tl.constexpr,
        softmax_scale2: tl.constexpr,
        quant_flag: tl.constexpr,
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
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < head_size
        queries = tl.load(
            query_ptr
            + query_tokens[:, None] * stride_query_token
            + query_heads[:, None] * stride_query_head
            + dimensions[None, :],
            mask=valid_query[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        kv_offsets = tl.arange(0, block_kv)
        max_score = tl.full([block_rows], -float("inf"), tl.float32)
        normalizer = tl.zeros([block_rows], tl.float32)
        accumulator = tl.zeros([block_rows, block_d], tl.float32)
        table_base = request_index * stride_block_table

        prefix_tiles = tl.cdiv(prefix_length, block_kv)
        for kv_tile in tl.range(0, prefix_tiles):
            logical_positions = kv_tile * block_kv + kv_offsets
            valid_kv = logical_positions < prefix_length
            logical_blocks = logical_positions // block_size
            block_offsets = logical_positions % block_size
            physical_blocks = tl.load(
                block_table_ptr + table_base + logical_blocks,
                mask=valid_kv,
                other=0,
            ).to(tl.int64)
            cache_bases = (
                physical_blocks * stride_cache_block + kv_head * stride_cache_head
            )
            cache_i16_bases = (
                physical_blocks * stride_cache_i16_block
                + kv_head * stride_cache_i16_head
            )
            map_bits = (
                tl.load(
                    cache_i16_ptr
                    + cache_i16_bases
                    + token_map_offset // 2
                    + block_offsets,
                    mask=valid_kv,
                    other=quant_flag,
                ).to(tl.int32)
                & 0xFFFF
            )
            is_anchor = map_bits < anchor_count

            if anchor_count:
                anchor_slots = tl.minimum(map_bits, anchor_count - 1)
                anchor_mask = (
                    valid_kv[:, None] & is_anchor[:, None] & dimension_mask[None, :]
                )
                anchor_key_bits = tl.load(
                    cache_i16_ptr
                    + cache_i16_bases[:, None]
                    + anchor_key_offset // 2
                    + anchor_slots[:, None] * head_size
                    + dimensions[None, :],
                    mask=anchor_mask,
                    other=0,
                )
                anchor_value_bits = tl.load(
                    cache_i16_ptr
                    + cache_i16_bases[:, None]
                    + anchor_value_offset // 2
                    + anchor_slots[:, None] * head_size
                    + dimensions[None, :],
                    mask=anchor_mask,
                    other=0,
                )
                anchor_key = anchor_key_bits.to(tl.bfloat16, bitcast=True)
                anchor_value = anchor_value_bits.to(tl.bfloat16, bitcast=True)

            if quant_count:
                quant_slots = tl.minimum(map_bits & (quant_flag - 1), quant_count - 1)
                quant_mask = (
                    valid_kv[:, None] & (~is_anchor[:, None]) & dimension_mask[None, :]
                )
                packed_key = tl.load(
                    cache_ptr
                    + cache_bases[:, None]
                    + quant_key_offset
                    + quant_slots[:, None] * packed_value_bytes
                    + dimensions[None, :] // 2,
                    mask=quant_mask,
                    other=0,
                ).to(tl.int32)
                packed_value = tl.load(
                    cache_ptr
                    + cache_bases[:, None]
                    + quant_value_offset
                    + quant_slots[:, None] * packed_value_bytes
                    + dimensions[None, :] // 2,
                    mask=quant_mask,
                    other=0,
                ).to(tl.int32)
                even_dimension = (dimensions[None, :] & 1) == 0
                nibble_key = tl.where(
                    even_dimension, packed_key & 0xF, (packed_key >> 4) & 0xF
                )
                nibble_value = tl.where(
                    even_dimension, packed_value & 0xF, (packed_value >> 4) & 0xF
                )
                signed_key = tl.where(nibble_key >= 8, nibble_key - 16, nibble_key).to(
                    tl.float32
                )
                signed_value = tl.where(
                    nibble_value >= 8, nibble_value - 16, nibble_value
                ).to(tl.float32)
                group_indices = dimensions[None, :] // group_size
                key_scale_bits = tl.load(
                    cache_i16_ptr
                    + cache_i16_bases[:, None]
                    + key_scale_offset // 2
                    + quant_slots[:, None] * num_groups
                    + group_indices,
                    mask=quant_mask,
                    other=0,
                )
                value_scale_bits = tl.load(
                    cache_i16_ptr
                    + cache_i16_bases[:, None]
                    + value_scale_offset // 2
                    + quant_slots[:, None] * num_groups
                    + group_indices,
                    mask=quant_mask,
                    other=0,
                )
                key_scale = key_scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
                value_scale = value_scale_bits.to(tl.float16, bitcast=True).to(
                    tl.float32
                )
                quant_key = (signed_key * key_scale).to(tl.bfloat16)
                quant_value = (signed_value * value_scale).to(tl.bfloat16)

            if anchor_count and quant_count:
                keys = tl.where(is_anchor[:, None], anchor_key, quant_key)
                values = tl.where(is_anchor[:, None], anchor_value, quant_value)
            elif anchor_count:
                keys = anchor_key
                values = anchor_value
            else:
                keys = quant_key
                values = quant_value

            scores = tl.dot(queries, tl.trans(keys)) * softmax_scale2
            score_mask = valid_query[:, None] & valid_kv[None, :]
            scores = tl.where(score_mask, scores, -float("inf"))
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
            weighted_values = tl.dot(probabilities.to(tl.bfloat16), values)
            accumulator = accumulator * old_factor[:, None] + weighted_values.to(
                tl.float32
            )
            max_score = new_max

        safe_normalizer = tl.maximum(normalizer, 1e-20)
        output_base = (
            query_tokens[:, None] * stride_output_token
            + query_heads[:, None] * stride_output_head
            + dimensions[None, :]
        )
        tl.store(
            output_ptr + output_base,
            accumulator / safe_normalizer[:, None],
            mask=valid_query[:, None] & dimension_mask[None, :],
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


def _get_launch_config(query_group_size: int, head_size: int) -> tuple[int, ...]:
    block_h = min(query_group_size, 4)
    block_m = 16 if block_h == 1 else 8
    block_rows = triton.next_power_of_2(max(16, block_m * block_h))
    block_kv = 64
    block_d = triton.next_power_of_2(head_size)
    num_warps = 8 if block_rows >= 32 or block_d > 128 else 4
    return block_m, block_h, block_rows, block_kv, block_d, num_warps


def warmup_hyquant_prefix_kernel(
    layout: HyQuantBlockLayout,
    num_kv_heads: int,
    num_query_heads: int,
    scale: float,
    device: torch.device,
) -> None:
    """Compile the direct-prefix specialization during model setup."""
    if not HAS_TRITON or device.type != "cuda":
        return
    key = (device.index, layout, num_kv_heads, num_query_heads, float(scale))
    if key in _WARMED_KERNELS:
        return
    try:
        cache = torch.zeros(
            1,
            num_kv_heads,
            layout.head_page_bytes,
            device=device,
            dtype=torch.uint8,
        )
        block_table = torch.zeros(1, 1, device=device, dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1], device=device, dtype=torch.int32)
        prefix_lens = torch.full(
            (1,), layout.block_size, device=device, dtype=torch.int32
        )
        # Triton specializes integer strides by alignment and treats a stride
        # of one as constexpr. Compile both aligned and unaligned LSE row
        # strides so arbitrary cache-hit suffix lengths cannot JIT at runtime.
        for query_len in (16, 3):
            query = torch.zeros(
                query_len,
                num_query_heads,
                layout.head_size,
                device=device,
                dtype=torch.bfloat16,
            )
            query_start_loc[1] = query_len
            output = torch.empty_like(query)
            lse = torch.empty(
                num_query_heads, query_len, device=device, dtype=torch.float32
            )
            _launch_hyquant_prefix_attention(
                query,
                cache,
                block_table,
                query_start_loc,
                prefix_lens,
                output,
                lse,
                layout,
                num_query_heads // num_kv_heads,
                scale,
                max_query_len=query_len,
            )
        torch.cuda.synchronize(device)
        _WARMED_KERNELS.add(key)
    except Exception:
        # Warmup is optional; a real unsupported launch reports its error.
        return


def _launch_hyquant_prefix_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefix_lens: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    layout: HyQuantBlockLayout,
    query_group_size: int,
    scale: float,
    max_query_len: int,
) -> bool:
    if triton_hyquant_prefix_groupdot(
        query,
        kv_cache,
        block_table,
        query_start_loc,
        prefix_lens,
        output,
        lse,
        layout,
        scale,
        max_query_len,
    ):
        return True
    block_m, block_h, block_rows, block_kv, block_d, num_warps = _get_launch_config(
        query_group_size, layout.head_size
    )
    num_requests = prefix_lens.shape[0]
    head_tiles = kv_cache.shape[1] * triton.cdiv(query_group_size, block_h)
    grid = (num_requests, triton.cdiv(max_query_len, block_m), head_tiles)
    cache_i16 = kv_cache.view(torch.int16)
    _hyquant_prefix_tiled_kernel[grid](
        query,
        kv_cache,
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
        cache_i16.stride(0),
        cache_i16.stride(1),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        lse.stride(1),
        query_group_size=query_group_size,
        head_size=layout.head_size,
        block_size=layout.block_size,
        anchor_count=layout.anchor_count,
        quant_count=layout.quant_count,
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
        block_m=block_m,
        block_h=block_h,
        block_rows=block_rows,
        block_kv=block_kv,
        block_d=block_d,
        softmax_scale2=scale * _LOG2E,
        quant_flag=HYQUANT_QUANT_FLAG,
        ln2=_LN2,
        num_warps=num_warps,
        num_stages=2,
    )
    return False


@torch.no_grad()
def triton_hyquant_prefix_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefix_lens: torch.Tensor,
    layout: HyQuantBlockLayout,
    scale: float,
    max_query_len: int,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return compact-prefix partial output and natural-log LSE.

    Queries are flattened as ``[tokens, query_heads, head_size]`` and grouped
    by ``query_start_loc``. Prefix lengths are per request and refer only to
    complete physical blocks.
    """
    global _LAST_STATUS
    if not HAS_TRITON or not query.is_cuda:
        _LAST_STATUS = "no_triton_or_cuda"
        return None
    if (
        query.ndim != 3
        or query.dtype != torch.bfloat16
        or kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 3
        or kv_cache.shape[2] != layout.head_page_bytes
        or query.shape[2] != layout.head_size
        or kv_cache.shape[1] <= 0
        or query.shape[1] % kv_cache.shape[1]
        or query.stride(-1) != 1
        or kv_cache.stride(-1) != 1
        or block_table.ndim != 2
        or query_start_loc.ndim != 1
        or prefix_lens.ndim != 1
        or query_start_loc.shape[0] != prefix_lens.shape[0] + 1
        or block_table.shape[0] < prefix_lens.shape[0]
        or query_start_loc.device != query.device
        or prefix_lens.device != query.device
        or block_table.device != query.device
        or query_start_loc.dtype != torch.int32
        or prefix_lens.dtype != torch.int32
        or max_query_len <= 0
    ):
        _LAST_STATUS = "shape"
        return None
    if (
        layout.block_size not in (16, 32, 64)
        or layout.head_size > 256
        or layout.head_size % 2
        or layout.group_size % 2
    ):
        _LAST_STATUS = "unsupported_shape"
        return None
    if output is None:
        output = torch.empty_like(query)
    if lse is None:
        lse = torch.empty(
            query.shape[1], query.shape[0], device=query.device, dtype=torch.float32
        )
    if (
        output.shape != query.shape
        or output.device != query.device
        or output.dtype != query.dtype
        or output.stride(-1) != 1
        or lse.shape != (query.shape[1], query.shape[0])
        or lse.device != query.device
        or lse.dtype != torch.float32
    ):
        _LAST_STATUS = "output_shape"
        return None
    if query.numel() == 0:
        output.zero_()
        lse.fill_(-float("inf"))
        _LAST_STATUS = "empty"
        return output, lse
    used_groupdot = _launch_hyquant_prefix_attention(
        query,
        kv_cache,
        block_table,
        query_start_loc,
        prefix_lens,
        output,
        lse,
        layout,
        query.shape[1] // kv_cache.shape[1],
        scale,
        max_query_len,
    )
    _LAST_STATUS = "triton_groupdot" if used_groupdot else "triton_tiled"
    return output, lse


def hyquant_prefix_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small dense reference used by unit tests and diagnostics."""
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("Reference prefix K/V must have matching [N, H, D] shapes")
    if query.ndim != 3:
        raise ValueError("Reference prefix query must have shape [T, H, D]")
    if query.shape[1] % key.shape[1]:
        raise ValueError("Reference query heads must be a multiple of KV heads")
    group_size = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(group_size, dim=1)
    expanded_value = value.repeat_interleave(group_size, dim=1)
    scores = torch.einsum("thd,nhd->thn", query.float(), expanded_key.float()) * scale
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("thn,nhd->thd", probabilities, expanded_value.float()).to(
        query.dtype
    )
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1).contiguous()
    return output, lse
