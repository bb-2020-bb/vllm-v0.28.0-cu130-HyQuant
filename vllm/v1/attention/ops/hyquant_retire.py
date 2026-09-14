"""CUDA-graph-safe periodic retirement for the HyQuant page layout.

Retirement is deliberately split into an anchor-selection launch and a page
packing launch.  The selection is shared by all KV heads in a logical block,
which matches :func:`select_important_token_indices`; the packing launch then
quantizes each KV head independently with the same token map.  Both launches
read the device-side sequence metadata and become no-ops away from a boundary,
so callers can keep them inside a captured decode graph.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.ops.hyquant_common import (
    HYQUANT_QUANT_FLAG,
    HyQuantBlockLayout,
)

_LAST_HYQUANT_RETIRE_STATUS = "not_run"
_WARMED_HYQUANT_RETIRE_KERNELS: set[tuple] = set()


def get_hyquant_retire_status() -> str:
    return _LAST_HYQUANT_RETIRE_STATUS


if HAS_TRITON:

    @triton.jit
    def _hyquant_select_retire_anchors_kernel(
        query_ptr,
        hot_key_ptr,
        seq_lens_ptr,
        prompt_lens_ptr,
        state_slots_ptr,
        anchor_ptr,
        stride_query_token,
        stride_query_head,
        stride_hot_slot,
        stride_hot_head,
        stride_hot_token,
        anchor_stride_slot,
        anchor_stride_block,
        num_kv_heads: tl.constexpr,
        query_group_size: tl.constexpr,
        head_size: tl.constexpr,
        block_size: tl.constexpr,
        anchor_count: tl.constexpr,
        window_size: tl.constexpr,
        retire_interval: tl.constexpr,
        hot_capacity: tl.constexpr,
        block_d: tl.constexpr,
    ):
        request_index = tl.program_id(0)
        retire_offset = tl.program_id(1)

        sequence_length = tl.load(seq_lens_ptr + request_index).to(tl.int64)
        prompt_length = tl.load(prompt_lens_ptr + request_index).to(tl.int64)
        state_slot = tl.load(state_slots_ptr + request_index).to(tl.int64)

        initial = tl.maximum(prompt_length - window_size, 0) // block_size * block_size
        generated = tl.maximum(sequence_length - prompt_length, 0)
        committed = initial + generated // retire_interval * retire_interval
        max_allowed = (
            tl.maximum(sequence_length - window_size, 0) // block_size * block_size
        )
        committed = tl.minimum(committed, max_allowed)

        previous_length = tl.maximum(sequence_length - 1, 0)
        previous_generated = tl.maximum(previous_length - prompt_length, 0)
        previous_committed = (
            initial + previous_generated // retire_interval * retire_interval
        )
        previous_max_allowed = (
            tl.maximum(previous_length - window_size, 0) // block_size * block_size
        )
        previous_committed = tl.minimum(previous_committed, previous_max_allowed)

        logical_block = previous_committed // block_size + retire_offset
        logical_start = logical_block * block_size
        active = (
            (state_slot >= 0)
            & (sequence_length > 0)
            & (committed > previous_committed)
            & (logical_start < committed)
        )
        if not active:
            return

        # Use safe addresses for masked loads.  This matters for padded graph
        # rows whose state slot is -1.
        safe_slot = tl.maximum(state_slot, 0)
        safe_logical_block = tl.maximum(logical_block, 0)
        token_offsets = tl.arange(0, block_size)
        logical_positions = logical_start + token_offsets
        token_valid = active & (logical_positions < committed)
        ring_positions = logical_positions % hot_capacity
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < head_size
        group_offsets = tl.arange(0, query_group_size)

        scores = tl.full([block_size], -float("inf"), tl.float32)
        for kv_head in range(num_kv_heads):
            query_heads = kv_head * query_group_size + group_offsets
            query_bases = (
                request_index * stride_query_token
                + query_heads[:, None] * stride_query_head
            )
            query_rows = tl.load(
                query_ptr + query_bases + dimensions[None, :],
                mask=active & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            query_reference = tl.sum(query_rows, axis=0) / query_group_size

            hot_bases = (
                safe_slot * stride_hot_slot
                + kv_head * stride_hot_head
                + ring_positions[:, None] * stride_hot_token
            )
            keys = tl.load(
                hot_key_ptr + hot_bases + dimensions[None, :],
                mask=token_valid[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            head_scores = tl.sum(keys * query_reference[None, :], axis=1).abs()
            scores = tl.maximum(
                scores, tl.where(token_valid, head_scores, -float("inf"))
            )

        # The temporary is indexed by the persistent request slot rather than
        # the local batch row, so it remains valid when the scheduler reorders
        # requests between decode steps.
        anchor_base = (
            safe_slot * anchor_stride_slot + safe_logical_block * anchor_stride_block
        )
        for ordinal in range(anchor_count):
            selected = tl.argmax(scores, axis=0)
            tl.store(
                anchor_ptr + anchor_base + ordinal,
                selected.to(tl.int16),
                mask=active,
            )
            scores = tl.where(
                token_offsets == selected,
                -float("inf"),
                scores,
            )

    @triton.jit
    def _hyquant_pack_retire_blocks_kernel(
        query_ptr,
        hot_key_ptr,
        hot_value_ptr,
        cache_ptr,
        cache_i16_ptr,
        block_table_ptr,
        seq_lens_ptr,
        prompt_lens_ptr,
        state_slots_ptr,
        stride_query_token,
        stride_query_head,
        stride_hot_slot,
        stride_hot_head,
        stride_hot_token,
        stride_cache_block,
        stride_cache_head,
        stride_block_table,
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
        quant_flag: tl.constexpr,
        block_d: tl.constexpr,
        pair_d: tl.constexpr,
    ):
        request_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        retire_offset = tl.program_id(2)

        sequence_length = tl.load(seq_lens_ptr + request_index).to(tl.int64)
        prompt_length = tl.load(prompt_lens_ptr + request_index).to(tl.int64)
        state_slot = tl.load(state_slots_ptr + request_index).to(tl.int64)

        initial = tl.maximum(prompt_length - window_size, 0) // block_size * block_size
        generated = tl.maximum(sequence_length - prompt_length, 0)
        committed = initial + generated // retire_interval * retire_interval
        max_allowed = (
            tl.maximum(sequence_length - window_size, 0) // block_size * block_size
        )
        committed = tl.minimum(committed, max_allowed)

        previous_length = tl.maximum(sequence_length - 1, 0)
        previous_generated = tl.maximum(previous_length - prompt_length, 0)
        previous_committed = (
            initial + previous_generated // retire_interval * retire_interval
        )
        previous_max_allowed = (
            tl.maximum(previous_length - window_size, 0) // block_size * block_size
        )
        previous_committed = tl.minimum(previous_committed, previous_max_allowed)

        logical_block = previous_committed // block_size + retire_offset
        logical_start = logical_block * block_size
        active = (
            (state_slot >= 0)
            & (sequence_length > 0)
            & (committed > previous_committed)
            & (logical_start < committed)
        )
        if not active:
            return
        safe_slot = tl.maximum(state_slot, 0)
        safe_logical_block = tl.maximum(logical_block, 0)

        token_offsets = tl.arange(0, block_size)
        logical_positions = logical_start + token_offsets
        token_valid = logical_positions < committed
        ring_positions = logical_positions % hot_capacity
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < head_size
        group_offsets = tl.arange(0, query_group_size)

        # Recompute the shared block-local importance scores in every KV-head
        # program.  This removes the second launch and avoids a cross-program
        # synchronization point; all programs use the same deterministic
        # reduction and therefore produce the same token map.
        scores = tl.full([block_size], -float("inf"), tl.float32)
        for select_kv_head in range(num_kv_heads):
            query_heads = select_kv_head * query_group_size + group_offsets
            query_bases = (
                request_index * stride_query_token
                + query_heads[:, None] * stride_query_head
            )
            query_rows = tl.load(
                query_ptr + query_bases + dimensions[None, :],
                mask=dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            query_reference = tl.sum(query_rows, axis=0) / query_group_size
            select_hot_bases = (
                safe_slot * stride_hot_slot
                + select_kv_head * stride_hot_head
                + ring_positions[:, None] * stride_hot_token
            )
            select_keys = tl.load(
                hot_key_ptr + select_hot_bases + dimensions[None, :],
                mask=token_valid[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            head_scores = tl.sum(select_keys * query_reference[None, :], axis=1).abs()
            scores = tl.maximum(
                scores, tl.where(token_valid, head_scores, -float("inf"))
            )

        anchor_ordinals = tl.full([block_size], -1, tl.int32)
        for ordinal in range(anchor_count):
            selected = tl.argmax(scores, axis=0)
            anchor_ordinals = tl.where(
                token_offsets == selected,
                ordinal,
                anchor_ordinals,
            )
            scores = tl.where(token_offsets == selected, -float("inf"), scores)

        physical_block = tl.load(
            block_table_ptr + request_index * stride_block_table + safe_logical_block,
            mask=active,
            other=0,
        ).to(tl.int64)
        page_base_bytes = (
            physical_block * stride_cache_block + kv_head * stride_cache_head
        )
        page_base_i16 = page_base_bytes // 2
        hot_bases = (
            safe_slot * stride_hot_slot
            + kv_head * stride_hot_head
            + ring_positions[:, None] * stride_hot_token
        )
        is_anchor = anchor_ordinals >= 0
        is_quantized = token_valid & ~is_anchor
        quant_ordinals = tl.cumsum(is_quantized.to(tl.int32), axis=0) - is_quantized.to(
            tl.int32
        )

        # Write the shared token map first.  The map is replicated per KV head
        # in the page, matching the existing vectorized packer.
        map_bits = tl.where(
            is_anchor,
            anchor_ordinals,
            quant_flag + quant_ordinals,
        ).to(tl.uint16)
        tl.store(
            cache_i16_ptr + page_base_i16 + token_map_offset // 2 + token_offsets,
            map_bits,
            mask=token_valid,
        )

        # Quantize each channel group independently.  Keeping the group tile
        # small avoids materializing a [block, head_size] FP32 temporary.
        group_offsets = tl.arange(0, pair_d)
        even_offsets = group_offsets * 2
        odd_offsets = even_offsets + 1
        for group_index in range(num_groups):
            dim_even = group_index * group_size + even_offsets
            dim_odd = group_index * group_size + odd_offsets
            even_mask = dim_even < head_size
            odd_mask = dim_odd < head_size
            key_even = tl.load(
                hot_key_ptr + hot_bases + dim_even[None, :],
                mask=token_valid[:, None] & even_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            key_odd = tl.load(
                hot_key_ptr + hot_bases + dim_odd[None, :],
                mask=token_valid[:, None] & odd_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            value_even = tl.load(
                hot_value_ptr + hot_bases + dim_even[None, :],
                mask=token_valid[:, None] & even_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            value_odd = tl.load(
                hot_value_ptr + hot_bases + dim_odd[None, :],
                mask=token_valid[:, None] & odd_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            key_amax = tl.maximum(
                tl.max(tl.abs(key_even), axis=1),
                tl.max(tl.abs(key_odd), axis=1),
            )
            value_amax = tl.maximum(
                tl.max(tl.abs(value_even), axis=1),
                tl.max(tl.abs(value_odd), axis=1),
            )
            key_scale = tl.where(key_amax > 0.0, key_amax / 7.0, 1.0)
            value_scale = tl.where(value_amax > 0.0, value_amax / 7.0, 1.0)

            key_even_q = tl.where(
                key_even == key_even,
                tl.extra.cuda.libdevice.round(key_even / key_scale[:, None]),
                0.0,
            ).to(tl.int32)
            key_odd_q = tl.where(
                key_odd == key_odd,
                tl.extra.cuda.libdevice.round(key_odd / key_scale[:, None]),
                0.0,
            ).to(tl.int32)
            value_even_q = tl.where(
                value_even == value_even,
                tl.extra.cuda.libdevice.round(value_even / value_scale[:, None]),
                0.0,
            ).to(tl.int32)
            value_odd_q = tl.where(
                value_odd == value_odd,
                tl.extra.cuda.libdevice.round(value_odd / value_scale[:, None]),
                0.0,
            ).to(tl.int32)
            key_even_q = tl.minimum(tl.maximum(key_even_q, -8), 7)
            key_odd_q = tl.minimum(tl.maximum(key_odd_q, -8), 7)
            value_even_q = tl.minimum(tl.maximum(value_even_q, -8), 7)
            value_odd_q = tl.minimum(tl.maximum(value_odd_q, -8), 7)
            packed_key = ((key_even_q & 0xF) | ((key_odd_q & 0xF) << 4)).to(tl.uint8)
            packed_value = ((value_even_q & 0xF) | ((value_odd_q & 0xF) << 4)).to(
                tl.uint8
            )
            quant_mask = is_quantized[:, None] & even_mask[None, :]
            quant_byte_offset = group_index * (group_size // 2)
            tl.store(
                cache_ptr
                + page_base_bytes
                + quant_key_offset
                + quant_ordinals[:, None] * packed_value_bytes
                + quant_byte_offset
                + group_offsets[None, :],
                packed_key,
                mask=quant_mask,
            )
            tl.store(
                cache_ptr
                + page_base_bytes
                + quant_value_offset
                + quant_ordinals[:, None] * packed_value_bytes
                + quant_byte_offset
                + group_offsets[None, :],
                packed_value,
                mask=quant_mask,
            )
            tl.store(
                cache_i16_ptr
                + page_base_i16
                + key_scale_offset // 2
                + quant_ordinals * num_groups
                + group_index,
                key_scale.to(tl.float16).to(tl.uint16, bitcast=True),
                mask=is_quantized,
            )
            tl.store(
                cache_i16_ptr
                + page_base_i16
                + value_scale_offset // 2
                + quant_ordinals * num_groups
                + group_index,
                value_scale.to(tl.float16).to(tl.uint16, bitcast=True),
                mask=is_quantized,
            )

        # Anchors remain BF16.  Load/store them as bit patterns so the result
        # is byte-identical to the existing torch packer.
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < head_size
        full_key = tl.load(
            hot_key_ptr + hot_bases + dimensions[None, :],
            mask=token_valid[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        full_value = tl.load(
            hot_value_ptr + hot_bases + dimensions[None, :],
            mask=token_valid[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        for ordinal in range(anchor_count):
            anchor_mask = token_valid & (anchor_ordinals == ordinal)
            selected_key = tl.sum(tl.where(anchor_mask[:, None], full_key, 0.0), axis=0)
            selected_value = tl.sum(
                tl.where(anchor_mask[:, None], full_value, 0.0), axis=0
            )
            tl.store(
                cache_i16_ptr
                + page_base_i16
                + anchor_key_offset // 2
                + ordinal * head_size
                + dimensions[None, :],
                selected_key[None, :].to(tl.uint16, bitcast=True),
                mask=active & dimension_mask[None, :],
            )
            tl.store(
                cache_i16_ptr
                + page_base_i16
                + anchor_value_offset // 2
                + ordinal * head_size
                + dimensions[None, :],
                selected_value[None, :].to(tl.uint16, bitcast=True),
                mask=active & dimension_mask[None, :],
            )


def triton_hyquant_retire_blocks(
    query: torch.Tensor,
    hot_key: torch.Tensor,
    hot_value: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    state_slots: torch.Tensor,
    anchor_workspace: torch.Tensor,
    layout: HyQuantBlockLayout,
    window_size: int,
    retire_interval: int,
    max_blocks: int,
) -> bool:
    """Run graph-safe retirement; return ``False`` when unsupported."""
    global _LAST_HYQUANT_RETIRE_STATUS
    if not HAS_TRITON or query.device.type != "cuda":
        _LAST_HYQUANT_RETIRE_STATUS = "requires_cuda"
        return False
    if (
        query.ndim != 3
        or hot_key.ndim != 4
        or hot_value.shape != hot_key.shape
        or query.shape[0] <= 0
        or query.shape[1] % hot_key.shape[1]
        or query.shape[2] != layout.head_size
        or hot_key.shape[2] <= 0
        or layout.block_size <= 0
        or layout.block_size % 2
        or layout.group_size <= 0
        or layout.group_size % 2
        or retire_interval <= 0
        or retire_interval % layout.block_size
        or max_blocks <= 0
    ):
        _LAST_HYQUANT_RETIRE_STATUS = "unsupported_shape"
        return False
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 3
        or kv_cache.shape[1] != hot_key.shape[1]
        or kv_cache.shape[2] != layout.head_page_bytes
        or block_table.ndim != 2
        or block_table.shape[0] < query.shape[0]
        or block_table.shape[1] < max_blocks
        or anchor_workspace.dtype != torch.int16
        or anchor_workspace.ndim != 3
        or anchor_workspace.shape[0] < int(hot_key.shape[0])
        or anchor_workspace.shape[1] < max_blocks
        or anchor_workspace.shape[2] < max(1, layout.anchor_count)
    ):
        _LAST_HYQUANT_RETIRE_STATUS = "unsupported_buffers"
        return False

    num_retire_blocks = retire_interval // layout.block_size
    cache_i16 = kv_cache.view(torch.int16)
    pair_d = layout.group_size // 2
    _hyquant_pack_retire_blocks_kernel[
        (query.shape[0], hot_key.shape[1], num_retire_blocks)
    ](
        query,
        hot_key,
        hot_value,
        kv_cache,
        cache_i16,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        query.stride(0),
        query.stride(1),
        hot_key.stride(0),
        hot_key.stride(1),
        hot_key.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        block_table.stride(0),
        num_kv_heads=hot_key.shape[1],
        query_group_size=query.shape[1] // hot_key.shape[1],
        head_size=layout.head_size,
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
        quant_flag=HYQUANT_QUANT_FLAG,
        block_d=triton.next_power_of_2(layout.head_size),
        pair_d=triton.next_power_of_2(pair_d),
        num_warps=8,
        num_stages=1,
    )
    _LAST_HYQUANT_RETIRE_STATUS = "gpu_graph_safe"
    return True


def warmup_hyquant_retire_kernels(
    layout: HyQuantBlockLayout,
    num_kv_heads: int,
    num_query_heads: int,
    window_size: int,
    retire_interval: int,
    hot_capacity: int,
    device: torch.device,
    max_blocks: int = 1,
) -> None:
    """Compile retirement specializations before graph capture/serving."""
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
        max_blocks,
    )
    if key in _WARMED_HYQUANT_RETIRE_KERNELS:
        return
    try:
        blocks = max(1, int(max_blocks))
        query = torch.zeros(
            1, num_query_heads, layout.head_size, dtype=torch.bfloat16, device=device
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
        cache = torch.zeros(
            blocks,
            num_kv_heads,
            layout.head_page_bytes,
            dtype=torch.uint8,
            device=device,
        )
        table = torch.zeros((1, blocks), dtype=torch.int32, device=device)
        seq = torch.ones((1,), dtype=torch.int32, device=device)
        prompt = torch.ones_like(seq)
        slots = torch.zeros_like(seq)
        anchors = torch.zeros(
            1,
            blocks,
            max(1, layout.anchor_count),
            dtype=torch.int16,
            device=device,
        )
        if triton_hyquant_retire_blocks(
            query,
            hot_key,
            hot_value,
            cache,
            table,
            seq,
            prompt,
            slots,
            anchors,
            layout,
            window_size,
            retire_interval,
            blocks,
        ):
            _WARMED_HYQUANT_RETIRE_KERNELS.add(key)
    except Exception:
        # Setup warmup is best effort; the wrapper still reports a precise
        # failure and the backend can use its host fallback.
        return
