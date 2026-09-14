"""BF16 hot-ring storage for the block-layout HyQuant backend."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

_LAST_HYQUANT_HOT_STATUS = "not_run"
_WARMED_HYQUANT_HOT_KERNELS: set[tuple] = set()
_WARMED_HYQUANT_DECODE_STORE_KERNELS: set[tuple] = set()


def get_hyquant_hot_status() -> str:
    return _LAST_HYQUANT_HOT_STATUS


def warmup_hyquant_hot_kernels(
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    window_size: int,
    retire_interval: int,
    hot_capacity: int,
    device: torch.device,
) -> None:
    """Compile both hot-store constexpr variants before the first request.

    vLLM's generic warmup does not necessarily exercise both the initial
    prefill and decode store signatures. Keeping this tiny, shape-keyed warmup
    here avoids charging the first user request for Triton compilation.
    """
    if not HAS_TRITON or device.type != "cuda":
        return
    key = (
        device.index,
        num_kv_heads,
        head_size,
        block_size,
        window_size,
        retire_interval,
        hot_capacity,
    )
    if key in _WARMED_HYQUANT_HOT_KERNELS:
        return
    try:
        source_key = torch.zeros(
            1, num_kv_heads, head_size, dtype=torch.bfloat16, device=device
        )
        source_value = torch.zeros_like(source_key)
        hot_key = torch.zeros(
            1,
            num_kv_heads,
            hot_capacity,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        )
        hot_value = torch.zeros_like(hot_key)
        # Use a valid row and a position in the hot window.  Invalid rows can
        # compile a different predicated path while hiding launch errors, and
        # then the first real request still pays the JIT cost.
        # The V1 block table exposes slot mappings and token positions as
        # int64; matching those dtypes is necessary because Triton includes
        # pointer element types in its specialization key.
        slot_mapping = torch.zeros((1,), dtype=torch.int64, device=device)
        positions = torch.full((1,), window_size, dtype=torch.int64, device=device)
        token_to_req = torch.zeros((1,), dtype=torch.int32, device=device)
        state_slots = torch.zeros((1,), dtype=torch.int32, device=device)
        seq_lens = torch.full((1,), window_size + 1, dtype=torch.int32, device=device)
        prompt_lens = torch.full((1,), window_size, dtype=torch.int32, device=device)
        cached_prefix_lens = torch.zeros((1,), dtype=torch.int32, device=device)
        for is_prefill in (False, True):
            _hyquant_hot_store_kernel[(1, num_kv_heads)](
                source_key,
                source_value,
                hot_key,
                hot_value,
                slot_mapping,
                positions,
                token_to_req,
                state_slots,
                seq_lens,
                prompt_lens,
                cached_prefix_lens,
                source_key.stride(0),
                source_key.stride(1),
                source_value.stride(0),
                source_value.stride(1),
                hot_key.stride(0),
                hot_key.stride(1),
                hot_key.stride(2),
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                block_size=block_size,
                window_size=window_size,
                retire_interval=retire_interval,
                hot_capacity=hot_capacity,
                block_d=triton.next_power_of_2(head_size),
                is_prefill=is_prefill,
                num_warps=4,
                num_stages=1,
            )
            # Triton compilation and the launch are asynchronous with respect
            # to the Python caller on some CUDA/Triton combinations.
            torch.cuda.synchronize(device)
        _WARMED_HYQUANT_HOT_KERNELS.add(key)
    except Exception:
        # A non-CUDA test or an unusual Triton target can still use the normal
        # lazy launch path; the warmup is an optimization, never a capability
        # requirement.
        return


if HAS_TRITON:
    # K/V tensors passed by fused QKV projections are often views whose token
    # stride is the full QKV width (rather than ``num_kv_heads * head_size``).
    # Keep address strides runtime-valued so the setup warmup covers both
    # contiguous and split-QKV layouts without a first-request recompilation.
    @triton.jit(
        do_not_specialize=[
            "stride_key_token",
            "stride_key_head",
            "stride_value_token",
            "stride_value_head",
            "stride_hot_slot",
            "stride_hot_head",
            "stride_hot_token",
        ],
        # QKV split views can have a different base-pointer alignment from
        # the setup tensors even when their strides match.  Alignment is not
        # required by the masked scalar/vector accesses in this kernel.
        do_not_specialize_on_alignment=[
            "key_ptr",
            "value_ptr",
            "hot_key_ptr",
            "hot_value_ptr",
            "slot_mapping_ptr",
            "positions_ptr",
            "token_to_req_ptr",
            "state_slots_ptr",
            "seq_lens_ptr",
            "prompt_lens_ptr",
            "cached_prefix_lens_ptr",
        ],
    )
    def _hyquant_hot_store_kernel(
        key_ptr,
        value_ptr,
        hot_key_ptr,
        hot_value_ptr,
        slot_mapping_ptr,
        positions_ptr,
        token_to_req_ptr,
        state_slots_ptr,
        seq_lens_ptr,
        prompt_lens_ptr,
        cached_prefix_lens_ptr,
        stride_key_token,
        stride_key_head,
        stride_value_token,
        stride_value_head,
        stride_hot_slot,
        stride_hot_head,
        stride_hot_token,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        block_size: tl.constexpr,
        window_size: tl.constexpr,
        retire_interval: tl.constexpr,
        hot_capacity: tl.constexpr,
        block_d: tl.constexpr,
        is_prefill: tl.constexpr,
    ):
        token_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        request_index = tl.load(token_to_req_ptr + token_index).to(tl.int64)
        state_slot = tl.load(state_slots_ptr + request_index).to(tl.int64)
        logical_position = tl.load(positions_ptr + token_index).to(tl.int64)
        sequence_length = tl.load(seq_lens_ptr + request_index).to(tl.int64)
        prompt_length = tl.load(prompt_lens_ptr + request_index).to(tl.int64)

        if is_prefill:
            window_boundary = (
                tl.maximum(sequence_length - window_size, 0) // block_size * block_size
            )
            cached_prefix = tl.load(cached_prefix_lens_ptr + request_index).to(tl.int64)
            committed = tl.maximum(window_boundary, cached_prefix)
        else:
            initial = (
                tl.maximum(prompt_length - window_size, 0) // block_size * block_size
            )
            generated = tl.maximum(sequence_length - prompt_length, 0)
            completed_intervals = generated // retire_interval
            committed = initial + completed_intervals * retire_interval
            max_allowed = (
                tl.maximum(sequence_length - window_size, 0) // block_size * block_size
            )
            committed = tl.minimum(committed, max_allowed)

        kv_slot = tl.load(slot_mapping_ptr + token_index)
        valid = (
            (state_slot >= 0)
            & (kv_slot >= 0)
            & (logical_position >= committed)
            & (logical_position < sequence_length)
        )
        ring_position = logical_position % hot_capacity
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < head_size
        source_key = (
            token_index * stride_key_token + kv_head * stride_key_head + dimensions
        )
        source_value = (
            token_index * stride_value_token + kv_head * stride_value_head + dimensions
        )
        hot_base = (
            state_slot * stride_hot_slot
            + kv_head * stride_hot_head
            + ring_position * stride_hot_token
            + dimensions
        )
        key = tl.load(
            key_ptr + source_key,
            mask=valid & dimension_mask,
            other=0.0,
        ).to(tl.bfloat16)
        value = tl.load(
            value_ptr + source_value,
            mask=valid & dimension_mask,
            other=0.0,
        ).to(tl.bfloat16)
        tl.store(hot_key_ptr + hot_base, key, mask=valid & dimension_mask)
        tl.store(hot_value_ptr + hot_base, value, mask=valid & dimension_mask)


    # Pure one-token decode has a much smaller ABI than prefill/extend.  The
    # scheduler already places decode rows in request order and seq_lens
    # includes the current token, so the generic slot/position/token-to-request
    # metadata is redundant here.  Keeping this kernel separate also avoids
    # carrying the prefill branch and its extra pointer arguments into the
    # hot path used by every decoder layer.
    @triton.jit(
        do_not_specialize=[
            "stride_key_token",
            "stride_key_head",
            "stride_value_token",
            "stride_value_head",
            "stride_hot_slot",
            "stride_hot_head",
            "stride_hot_token",
        ],
        do_not_specialize_on_alignment=[
            "key_ptr",
            "value_ptr",
            "hot_key_ptr",
            "hot_value_ptr",
            "state_slots_ptr",
            "seq_lens_ptr",
            "prompt_lens_ptr",
        ],
    )
    def _hyquant_decode_store_current_kernel(
        key_ptr,
        value_ptr,
        hot_key_ptr,
        hot_value_ptr,
        state_slots_ptr,
        seq_lens_ptr,
        prompt_lens_ptr,
        stride_key_token,
        stride_key_head,
        stride_value_token,
        stride_value_head,
        stride_hot_slot,
        stride_hot_head,
        stride_hot_token,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        block_size: tl.constexpr,
        window_size: tl.constexpr,
        retire_interval: tl.constexpr,
        hot_capacity: tl.constexpr,
        block_d: tl.constexpr,
    ):
        request_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        state_slot = tl.load(state_slots_ptr + request_index).to(tl.int64)
        sequence_length = tl.load(seq_lens_ptr + request_index).to(tl.int64)
        prompt_length = tl.load(prompt_lens_ptr + request_index).to(tl.int64)
        logical_position = sequence_length - 1
        initial = (
            tl.maximum(prompt_length - window_size, 0) // block_size * block_size
        )
        generated = tl.maximum(sequence_length - prompt_length, 0)
        committed = initial + (generated // retire_interval) * retire_interval
        max_allowed = (
            tl.maximum(sequence_length - window_size, 0) // block_size * block_size
        )
        committed = tl.minimum(committed, max_allowed)
        valid = (state_slot >= 0) & (logical_position >= committed)
        ring_position = logical_position % hot_capacity
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < head_size
        source_base = (
            request_index * stride_key_token
            + kv_head * stride_key_head
            + dimensions
        )
        value_source_base = (
            request_index * stride_value_token
            + kv_head * stride_value_head
            + dimensions
        )
        hot_base = (
            state_slot * stride_hot_slot
            + kv_head * stride_hot_head
            + ring_position * stride_hot_token
            + dimensions
        )
        mask = valid & dimension_mask
        key = tl.load(key_ptr + source_base, mask=mask, other=0.0).to(tl.bfloat16)
        value = tl.load(
            value_ptr + value_source_base, mask=mask, other=0.0
        ).to(tl.bfloat16)
        tl.store(hot_key_ptr + hot_base, key, mask=mask)
        tl.store(hot_value_ptr + hot_base, value, mask=mask)


def triton_hyquant_store_hot(
    key: torch.Tensor,
    value: torch.Tensor,
    hot_key: torch.Tensor,
    hot_value: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    token_to_req: torch.Tensor,
    state_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    block_size: int,
    window_size: int,
    retire_interval: int,
    *,
    cached_prefix_lens: torch.Tensor | None = None,
    is_prefill: bool = False,
) -> bool:
    """Append current BF16 K/V rows to each request's circular hot window."""
    global _LAST_HYQUANT_HOT_STATUS
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("HyQuant hot-store K/V must have shape [tokens, heads, dim]")
    if key.shape[0] != positions.numel() or key.shape[0] != token_to_req.numel():
        raise ValueError("HyQuant hot-store token metadata has inconsistent length")
    if not HAS_TRITON or not key.is_cuda:
        _LAST_HYQUANT_HOT_STATUS = "requires_cuda"
        return False
    if is_prefill and cached_prefix_lens is None:
        raise ValueError("HyQuant prefill hot store requires cached-prefix lengths")
    if cached_prefix_lens is None:
        cached_prefix_lens = prompt_lens
    if hot_key.dtype != torch.bfloat16 or hot_value.dtype != torch.bfloat16:
        raise ValueError("HyQuant hot ring must use BF16 storage")
    if key.shape[1] != hot_key.shape[1] or key.shape[-1] != hot_key.shape[-1]:
        raise ValueError("HyQuant hot ring shape does not match current K/V")
    if key.shape[0] == 0:
        _LAST_HYQUANT_HOT_STATUS = "empty"
        return True

    _hyquant_hot_store_kernel[(key.shape[0], key.shape[1])](
        key,
        value,
        hot_key,
        hot_value,
        slot_mapping,
        positions,
        token_to_req,
        state_slots,
        seq_lens,
        prompt_lens,
        cached_prefix_lens,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        hot_key.stride(0),
        hot_key.stride(1),
        hot_key.stride(2),
        num_kv_heads=key.shape[1],
        head_size=key.shape[-1],
        block_size=block_size,
        window_size=window_size,
        retire_interval=retire_interval,
        hot_capacity=hot_key.shape[2],
        block_d=triton.next_power_of_2(key.shape[-1]),
        is_prefill=is_prefill,
        num_warps=4,
        num_stages=1,
    )
    _LAST_HYQUANT_HOT_STATUS = "bf16_hot"
    return True


def triton_hyquant_store_decode_current(
    key: torch.Tensor,
    value: torch.Tensor,
    hot_key: torch.Tensor,
    hot_value: torch.Tensor,
    state_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    block_size: int,
    window_size: int,
    retire_interval: int,
) -> bool:
    """Store one current decode row per request using the compact ABI.

    This is intentionally restricted to the scheduler's uniform one-token
    decode shape.  The general :func:`triton_hyquant_store_hot` remains the
    implementation for prefill and multi-token extends, where logical
    positions and request mappings cannot be inferred from ``seq_lens`` alone.
    """
    global _LAST_HYQUANT_HOT_STATUS
    if not HAS_TRITON or not key.is_cuda:
        _LAST_HYQUANT_HOT_STATUS = "requires_cuda"
        return False
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("decode hot-store K/V must have shape [batch, kv_heads, dim]")
    batch_size, num_kv_heads, head_size = key.shape
    if (
        hot_key.ndim != 4
        or hot_value.shape != hot_key.shape
        or hot_key.shape[0] < batch_size
        or hot_key.shape[1] != num_kv_heads
        or hot_key.shape[3] != head_size
        or hot_key.dtype != torch.bfloat16
        or hot_value.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise ValueError("decode hot-store tensors have incompatible shapes/dtypes")
    if (
        state_slots.ndim != 1
        or seq_lens.shape != state_slots.shape
        or prompt_lens.shape != state_slots.shape
        or state_slots.numel() < batch_size
        or state_slots.device != key.device
        or seq_lens.device != key.device
        or prompt_lens.device != key.device
    ):
        raise ValueError("decode hot-store request metadata has an invalid shape")
    if key.shape[0] == 0:
        _LAST_HYQUANT_HOT_STATUS = "empty"
        return True
    _hyquant_decode_store_current_kernel[(batch_size, num_kv_heads)](
        key,
        value,
        hot_key,
        hot_value,
        state_slots,
        seq_lens,
        prompt_lens,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        hot_key.stride(0),
        hot_key.stride(1),
        hot_key.stride(2),
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        window_size=window_size,
        retire_interval=retire_interval,
        hot_capacity=hot_key.shape[2],
        block_d=triton.next_power_of_2(head_size),
        num_warps=4,
        num_stages=1,
    )
    _LAST_HYQUANT_HOT_STATUS = "bf16_decode_current"
    return True


def warmup_hyquant_decode_store_kernel(
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    window_size: int,
    retire_interval: int,
    hot_capacity: int,
    device: torch.device,
) -> None:
    """Compile the uniform decode store specialization during setup."""
    if not HAS_TRITON or device.type != "cuda":
        return
    key = (
        device.index,
        num_kv_heads,
        head_size,
        block_size,
        window_size,
        retire_interval,
        hot_capacity,
    )
    if key in _WARMED_HYQUANT_DECODE_STORE_KERNELS:
        return
    try:
        source_key = torch.zeros(
            1, num_kv_heads, head_size, dtype=torch.bfloat16, device=device
        )
        source_value = torch.zeros_like(source_key)
        hot_key = torch.zeros(
            1, num_kv_heads, hot_capacity, head_size,
            dtype=torch.bfloat16, device=device,
        )
        hot_value = torch.zeros_like(hot_key)
        state_slots = torch.zeros(1, dtype=torch.int32, device=device)
        seq_lens = torch.full(
            (1,), window_size + 1, dtype=torch.int32, device=device
        )
        prompt_lens = torch.full(
            (1,), window_size, dtype=torch.int32, device=device
        )
        triton_hyquant_store_decode_current(
            source_key,
            source_value,
            hot_key,
            hot_value,
            state_slots,
            seq_lens,
            prompt_lens,
            block_size,
            window_size,
            retire_interval,
        )
        torch.cuda.synchronize(device)
        _WARMED_HYQUANT_DECODE_STORE_KERNELS.add(key)
    except Exception:
        return
