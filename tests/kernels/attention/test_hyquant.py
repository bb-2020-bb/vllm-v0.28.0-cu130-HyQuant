"""Tests for the compact mixed-token HyQuant page format."""

from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.attention.backends.hyquant_attn import HyQuantAttentionImpl
from vllm.v1.attention.ops.hyquant_block import (
    gather_hyquant_block_kv,
    get_hyquant_block_decode_status,
    hyquant_block_decode_attention,
    materialize_hyquant_blocks,
    pack_hyquant_blocks,
    select_important_token_indices,
    triton_hyquant_block_decode_attention,
)
from vllm.v1.attention.ops.hyquant_common import (
    HYQUANT_QUANT_FLAG,
    HyQuantBlockLayout,
    get_hyquant_committed_len,
    get_hyquant_hot_capacity,
    make_token_map,
    symmetric_int4_groupwise,
    unpack_int4_groupwise,
)
from vllm.v1.attention.ops.hyquant_decode_groupdot import (
    get_hyquant_decode_split_count,
)
from vllm.v1.attention.ops.hyquant_decode_int8 import (
    get_hyquant_int8qk_warmup_configs,
)
from vllm.v1.attention.ops.hyquant_hot import triton_hyquant_store_hot
from vllm.v1.attention.ops.hyquant_prefill import (
    get_hyquant_prefix_status,
    hyquant_prefix_attention_reference,
    triton_hyquant_prefix_attention,
)


def test_compact_layout_has_real_memory_reduction() -> None:
    bf16_bytes = 16 * 128 * 4
    layout = HyQuantBlockLayout.from_params(16, 128, 32, 1 / 16)
    assert layout.anchor_count == 1
    assert layout.quant_count == 15
    assert layout.head_page_bytes < bf16_bytes
    assert layout.compact_bytes_per_token < 0.4 * 512
    assert layout.quant_scales_per_block == 15 * 4


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_layout_page_size_matches_allocator_slot_size(block_size: int) -> None:
    for head_size in (64, 80, 96, 128, 192, 256):
        layout = HyQuantBlockLayout.from_params(
            block_size, head_size, group_size=32, top_ratio=1 / 16
        )
        assert layout.head_page_bytes == block_size * layout.bytes_per_token


@pytest.mark.parametrize("group_size", [16, 32, 64, 256])
def test_groupwise_round_trip(group_size: int) -> None:
    torch.manual_seed(7)
    values = torch.randn(3, 16, 128)
    packed, scales = symmetric_int4_groupwise(values, group_size)
    restored = unpack_int4_groupwise(packed, scales, 128, group_size)
    assert scales.shape[-1] == math.ceil(128 / group_size)
    assert packed.shape[-1] == 64
    assert torch.isfinite(restored).all()
    assert (restored - values).abs().mean() < 0.12


def test_token_map_has_anchor_and_quant_slots() -> None:
    anchors = torch.tensor([[1, 5], [0, 7]], dtype=torch.int16)
    token_map, quant_indices = make_token_map(anchors, 8)
    assert token_map.dtype == torch.uint16
    assert quant_indices.shape == (2, 6)
    assert int(token_map[0, 1]) == 0
    assert int(token_map[0, 5]) == 1
    assert int(token_map[0, 0]) == HYQUANT_QUANT_FLAG


def test_token_selector_ratio_boundaries() -> None:
    key = torch.randn(32, 2, 8)
    query = torch.randn(4, 8)
    assert select_important_token_indices(key, query, 16, 0).shape == (2, 0)
    assert select_important_token_indices(key, query, 16, 1).shape == (2, 16)
    selected = select_important_token_indices(key, query, 16, 0.25)
    assert selected.shape == (2, 4)
    assert torch.all(selected[:, 1:] > selected[:, :-1])


def test_token_selector_accepts_prefix_stable_per_block_queries() -> None:
    torch.manual_seed(11)
    key = torch.randn(32, 2, 8)
    block_queries = torch.randn(2, 4, 8)
    selected = select_important_token_indices(key, block_queries, 16, 0.25)
    expected = torch.cat(
        [
            select_important_token_indices(
                key[index * 16 : (index + 1) * 16],
                block_queries[index],
                16,
                0.25,
            )
            for index in range(2)
        ]
    )
    assert torch.equal(selected, expected)


@pytest.mark.parametrize(
    ("sequence_length", "prompt_length", "expected"),
    [
        (256, 256, 0),
        (512, 512, 256),
        (576, 512, 320),
        (580, 512, 320),
        (640, 512, 384),
    ],
)
def test_retirement_boundary(
    sequence_length: int, prompt_length: int, expected: int
) -> None:
    assert (
        get_hyquant_committed_len(sequence_length, prompt_length, 16, 256, 64)
        == expected
    )


def test_int8qk_warmup_configs_cover_reachable_single_request_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "VLLM_HYQUANT_DECODE_SPLITS",
        "VLLM_HYQUANT_DECODE_BLOCK_KV",
        "VLLM_HYQUANT_DECODE_BLOCK_H",
        "VLLM_HYQUANT_DECODE_WARPS",
        "VLLM_HYQUANT_DECODE_STAGES",
    ):
        monkeypatch.delenv(name, raising=False)
    assert get_hyquant_int8qk_warmup_configs(4128, 1) == (
        (1, 1, 64, 4, 4, 2),
        (512, 4, 64, 4, 4, 2),
        (2048, 16, 128, 4, 4, 2),
    )


def test_int8qk_warmup_configs_remove_batch_only_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "VLLM_HYQUANT_DECODE_SPLITS",
        "VLLM_HYQUANT_DECODE_BLOCK_KV",
        "VLLM_HYQUANT_DECODE_BLOCK_H",
        "VLLM_HYQUANT_DECODE_WARPS",
        "VLLM_HYQUANT_DECODE_STAGES",
    ):
        monkeypatch.delenv(name, raising=False)
    configs = get_hyquant_int8qk_warmup_configs(32768, 32)
    assert {config[1:] for config in configs} == {
        (1, 64, 4, 4, 2),
        (4, 64, 4, 4, 2),
        (8, 128, 4, 4, 2),
        (16, 128, 4, 4, 2),
        (32, 128, 4, 4, 2),
        (32, 128, 4, 4, 3),
        (64, 128, 4, 4, 3),
        (1, 32, 4, 4, 2),
        (8, 32, 4, 4, 2),
        (16, 32, 4, 4, 2),
    }


def test_single_request_long_decode_uses_higher_occupancy_split() -> None:
    assert get_hyquant_decode_split_count(1, 8192, 4) == 64
    assert get_hyquant_decode_split_count(2, 8192, 4) == 32
    assert get_hyquant_decode_split_count(3, 8192, 4) == 32


def test_dense_prefill_fallback_supports_gqa_and_mqa() -> None:
    impl = object.__new__(HyQuantAttentionImpl)
    impl.scale = 0.5
    tokens, query_heads, kv_heads, head_size = 5, 4, 2, 8
    query = torch.randn(tokens, query_heads, head_size, dtype=torch.bfloat16)
    key = torch.randn(tokens, kv_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    actual = impl._run_dense_prefill_fallback(query, key, value, output, [0, tokens])
    expanded_key = key.repeat_interleave(query_heads // kv_heads, dim=1)
    expanded_value = value.repeat_interleave(query_heads // kv_heads, dim=1)
    causal = torch.ones(tokens, tokens, dtype=torch.bool).tril()
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            expanded_key.transpose(0, 1).unsqueeze(0),
            expanded_value.transpose(0, 1).unsqueeze(0),
            attn_mask=causal,
            dropout_p=0.0,
            scale=impl.scale,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    assert torch.equal(actual, expected)


def test_dense_prefill_fallback_supports_rectangular_chunks() -> None:
    impl = object.__new__(HyQuantAttentionImpl)
    impl.scale = 0.25
    query_heads, kv_heads, head_size = 4, 2, 8
    query_lens = [2, 1]
    key_lens = [5, 3]
    query_starts = [0, 2, 3]
    key_starts = [0, 5, 8]
    query = torch.randn(3, query_heads, head_size, dtype=torch.bfloat16)
    key = torch.randn(8, kv_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = torch.empty_like(query)

    actual = impl._run_dense_prefill_fallback(
        query,
        key,
        value,
        output,
        query_starts,
        key_starts,
    )
    expected = torch.empty_like(query)
    for request_index, (query_len, key_len) in enumerate(
        zip(query_lens, key_lens, strict=True)
    ):
        q_start = query_starts[request_index]
        q_end = query_starts[request_index + 1]
        k_start = key_starts[request_index]
        k_end = key_starts[request_index + 1]
        expanded_key = key[k_start:k_end].repeat_interleave(2, dim=1)
        expanded_value = value[k_start:k_end].repeat_interleave(2, dim=1)
        context_len = key_len - query_len
        mask = torch.arange(key_len)[None, :] <= (
            context_len + torch.arange(query_len)[:, None]
        )
        request_output = torch.nn.functional.scaled_dot_product_attention(
            query[q_start:q_end].transpose(0, 1).unsqueeze(0),
            expanded_key.transpose(0, 1).unsqueeze(0),
            expanded_value.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            dropout_p=0.0,
            scale=impl.scale,
        )
        expected[q_start:q_end] = request_output.squeeze(0).transpose(0, 1)
    assert torch.equal(actual, expected)


def _make_staging_only_impl() -> HyQuantAttentionImpl:
    impl = object.__new__(HyQuantAttentionImpl)
    impl._prefill_staging = {}
    impl.block_size = 16
    return impl


def test_chunked_prefill_stages_uneven_non_aligned_chunks() -> None:
    impl = _make_staging_only_impl()
    full_key = torch.randn(11, 2, 8, dtype=torch.bfloat16)
    full_value = torch.randn_like(full_key)

    for start, end in ((0, 3), (3, 7)):
        keys, values, completed, completed_slots = impl._prepare_chunked_prefill_kv(
            full_key[start:end],
            full_value[start:end],
            [0, end - start],
            [end],
            [11],
            [0],
            [5],
        )
        assert torch.equal(keys[0], full_key[:end])
        assert torch.equal(values[0], full_value[:end])
        assert completed == []
        assert completed_slots == []
        assert impl._prefill_staging[5].filled_len == end

    keys, values, completed, completed_slots = impl._prepare_chunked_prefill_kv(
        full_key[7:],
        full_value[7:],
        [0, 4],
        [11],
        [11],
        [0],
        [5],
    )
    assert torch.equal(keys[0], full_key)
    assert torch.equal(values[0], full_value)
    assert completed == [0]
    assert completed_slots == [5]
    impl._release_completed_prefills(completed_slots)
    assert 5 not in impl._prefill_staging


def test_chunked_prefill_tracks_reordered_multi_request_slots() -> None:
    impl = _make_staging_only_impl()
    full_keys = [
        torch.randn(7, 2, 8, dtype=torch.bfloat16),
        torch.randn(9, 2, 8, dtype=torch.bfloat16),
    ]
    full_values = [torch.randn_like(item) for item in full_keys]

    impl._prepare_chunked_prefill_kv(
        torch.cat((full_keys[0][:3], full_keys[1][:4])),
        torch.cat((full_values[0][:3], full_values[1][:4])),
        [0, 3, 7],
        [3, 4],
        [7, 9],
        [0, 0],
        [3, 9],
    )
    keys, values, completed, completed_slots = impl._prepare_chunked_prefill_kv(
        torch.cat((full_keys[1][4:], full_keys[0][3:])),
        torch.cat((full_values[1][4:], full_values[0][3:])),
        [0, 5, 9],
        [9, 7],
        [9, 7],
        [0, 0],
        [9, 3],
    )
    assert torch.equal(keys[0], full_keys[1])
    assert torch.equal(values[0], full_values[1])
    assert torch.equal(keys[1], full_keys[0])
    assert torch.equal(values[1], full_values[0])
    assert completed == [0, 1]
    assert completed_slots == [9, 3]


def test_chunked_prefill_context_zero_replaces_reused_slot() -> None:
    impl = _make_staging_only_impl()
    old_key = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    old_value = torch.randn_like(old_key)
    impl._prepare_chunked_prefill_kv(old_key, old_value, [0, 2], [2], [8], [0], [4])

    new_key = torch.randn(3, 1, 8, dtype=torch.bfloat16)
    new_value = torch.randn_like(new_key)
    keys, values, _, _ = impl._prepare_chunked_prefill_kv(
        new_key, new_value, [0, 3], [3], [6], [0], [4]
    )
    assert impl._prefill_staging[4].prompt_len == 6
    assert impl._prefill_staging[4].filled_len == 3
    assert torch.equal(keys[0], new_key)
    assert torch.equal(values[0], new_value)


def test_chunked_prefill_stages_only_uncached_suffix() -> None:
    impl = _make_staging_only_impl()
    full_key = torch.randn(8, 1, 8, dtype=torch.bfloat16)
    full_value = torch.randn_like(full_key)

    keys, values, completed, completed_slots = impl._prepare_chunked_prefill_kv(
        full_key[:3],
        full_value[:3],
        [0, 3],
        [19],
        [24],
        [16],
        [6],
    )
    state = impl._prefill_staging[6]
    assert state.key.shape[0] == 8
    assert state.cached_prefix_len == 16
    assert state.filled_len == 19
    assert torch.equal(keys[0], full_key[:3])
    assert torch.equal(values[0], full_value[:3])
    assert completed == []
    assert completed_slots == []

    keys, values, completed, completed_slots = impl._prepare_chunked_prefill_kv(
        full_key[3:],
        full_value[3:],
        [0, 5],
        [24],
        [24],
        [16],
        [6],
    )
    assert torch.equal(keys[0], full_key)
    assert torch.equal(values[0], full_value)
    assert completed == [0]
    assert completed_slots == [6]


@pytest.mark.parametrize(
    ("setup", "query_len", "seq_len", "prompt_len", "cached_prefix", "match"),
    [
        (None, 2, 4, 8, 0, "without its earlier chunks"),
        (2, 2, 6, 8, 0, "expected offset 2, got 4"),
        (2, 2, 4, 9, 0, "prompt length changed"),
        (None, 2, 2, 8, 16, "cached prefix exceeds the prefill context"),
        (None, 2, 6, 8, 3, "must contain complete physical blocks"),
    ],
)
def test_chunked_prefill_rejects_invalid_continuations(
    setup: int | None,
    query_len: int,
    seq_len: int,
    prompt_len: int,
    cached_prefix: int,
    match: str,
) -> None:
    impl = _make_staging_only_impl()
    if setup is not None:
        initial_key = torch.randn(setup, 1, 8, dtype=torch.bfloat16)
        impl._prepare_chunked_prefill_kv(
            initial_key,
            torch.randn_like(initial_key),
            [0, setup],
            [setup],
            [8],
            [0],
            [2],
        )
    current_key = torch.randn(query_len, 1, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=match):
        impl._prepare_chunked_prefill_kv(
            current_key,
            torch.randn_like(current_key),
            [0, query_len],
            [seq_len],
            [prompt_len],
            [cached_prefix],
            [2],
        )


def _cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
    except Exception:
        return False
    return True


requires_cuda = pytest.mark.skipif(
    not _cuda_available(), reason="a usable CUDA device is required"
)


@requires_cuda
@pytest.mark.parametrize("ratio", [0.0, 1 / 16, 1.0])
def test_mixed_page_pack_and_gather(ratio: float) -> None:
    device = torch.device("cuda")
    block_size, num_blocks, num_heads, head_size = 16, 32, 2, 128
    layout = HyQuantBlockLayout.from_params(block_size, head_size, 32, ratio)
    key = torch.randn(
        num_blocks,
        block_size,
        num_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    query = torch.randn(num_heads, head_size, device=device)
    anchors = select_important_token_indices(
        key.reshape(-1, num_heads, head_size), query, block_size, ratio
    )
    cache = torch.zeros(
        num_blocks, num_heads, layout.head_page_bytes, dtype=torch.uint8, device=device
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)
    pack_hyquant_blocks(key, value, block_table, anchors, cache, layout)

    hot_key = key.reshape(-1, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
    hot_value = value.reshape(-1, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
    gathered_key, gathered_value = gather_hyquant_block_kv(
        cache,
        hot_key,
        hot_value,
        block_table,
        num_blocks * block_size,
        num_blocks * block_size,
        0,
        layout,
        256,
        64,
    )
    expected_key = key.reshape(-1, num_heads, head_size).float()
    expected_value = value.reshape(-1, num_heads, head_size).float()
    committed = get_hyquant_committed_len(
        num_blocks * block_size, num_blocks * block_size, block_size, 256, 64
    )
    assert torch.equal(gathered_key[committed:], expected_key[committed:])
    assert torch.equal(gathered_value[committed:], expected_value[committed:])
    if ratio == 1.0:
        assert torch.equal(gathered_key[:committed], expected_key[:committed])
        assert torch.equal(gathered_value[:committed], expected_value[:committed])
    elif ratio == 0.0:
        assert (gathered_key[:committed] - expected_key[:committed]).abs().mean() < 0.13
        assert (
            gathered_value[:committed] - expected_value[:committed]
        ).abs().mean() < 0.13


@requires_cuda
@pytest.mark.parametrize("ratio", [0.0, 1 / 16, 1.0])
def test_compact_page_materialization_preserves_logical_order(ratio: float) -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    block_size, num_blocks, num_heads, head_size = 16, 4, 2, 32
    layout = HyQuantBlockLayout.from_params(block_size, head_size, 16, ratio)
    key = torch.randn(
        num_blocks,
        block_size,
        num_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    block_queries = torch.randn(
        num_blocks, 4, head_size, dtype=torch.bfloat16, device=device
    )
    anchors = select_important_token_indices(
        key.reshape(-1, num_heads, head_size),
        block_queries,
        block_size,
        ratio,
    )
    cache = torch.zeros(
        num_blocks, num_heads, layout.head_page_bytes, dtype=torch.uint8, device=device
    )
    physical_ids = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
    pack_hyquant_blocks(key, value, physical_ids, anchors, cache, layout)

    actual_key, actual_value = materialize_hyquant_blocks(cache, physical_ids, layout)
    expected_key = key.reshape_as(actual_key)
    expected_value = value.reshape_as(actual_value)
    if ratio == 1.0:
        assert torch.equal(actual_key, expected_key)
        assert torch.equal(actual_value, expected_value)
    else:
        assert (actual_key.float() - expected_key.float()).abs().mean() < 0.13
        assert (actual_value.float() - expected_value.float()).abs().mean() < 0.13


def _make_prefill_pack_impl(
    block_size: int = 16,
    window_size: int = 16,
    num_kv_heads: int = 2,
    num_query_heads: int = 8,
    head_size: int = 32,
    top_ratio: float = 0.25,
) -> HyQuantAttentionImpl:
    impl = object.__new__(HyQuantAttentionImpl)
    impl._prefill_staging = {}
    impl.block_size = block_size
    impl.window_size = window_size
    impl.num_kv_heads = num_kv_heads
    impl.num_heads = num_query_heads
    impl.head_size = head_size
    impl.top_ratio = top_ratio
    impl.block_layout = HyQuantBlockLayout.from_params(
        block_size, head_size, 16, top_ratio
    )
    return impl


def _pack_prefill_at_boundaries(
    impl: HyQuantAttentionImpl,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    boundaries: tuple[int, ...],
    block_table: torch.Tensor,
    cache: torch.Tensor,
    cached_prefix: int = 0,
) -> None:
    context = cached_prefix
    prompt_length = key.shape[0]
    for sequence_length in boundaries:
        query_length = sequence_length - context
        rows = impl._validate_prefill_rows(
            [0, query_length],
            [sequence_length],
            [prompt_length],
            [cached_prefix],
            [3],
        )
        suffix_keys, suffix_values, _, completed_slots = (
            impl._prepare_chunked_prefill_kv(
                key[context:sequence_length],
                value[context:sequence_length],
                [0, query_length],
                [sequence_length],
                [prompt_length],
                [cached_prefix],
                [3],
                rows=rows,
            )
        )
        impl._pack_new_prefill_blocks(
            suffix_keys,
            suffix_values,
            query[context:sequence_length],
            [0, query_length],
            rows,
            block_table,
            cache,
        )
        impl._release_completed_prefills(completed_slots)
        context = sequence_length


@requires_cuda
def test_prefill_pages_are_stable_across_chunks_and_later_suffixes() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    impl = _make_prefill_pack_impl()
    prompt_length = 80
    key = torch.randn(prompt_length, 2, 32, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    query = torch.randn(prompt_length, 8, 32, dtype=torch.bfloat16, device=device)
    table = torch.arange(5, dtype=torch.int32, device=device).unsqueeze(0)
    one_shot_cache = torch.zeros(
        5,
        2,
        impl.block_layout.head_page_bytes,
        dtype=torch.uint8,
        device=device,
    )
    chunked_cache = torch.zeros_like(one_shot_cache)
    _pack_prefill_at_boundaries(
        impl, key, value, query, (prompt_length,), table, one_shot_cache
    )
    impl._prefill_staging.clear()
    _pack_prefill_at_boundaries(
        impl, key, value, query, (7, 19, 33, 58, prompt_length), table, chunked_cache
    )
    assert torch.equal(one_shot_cache[:4], chunked_cache[:4])

    changed_key = key.clone()
    changed_value = value.clone()
    changed_query = query.clone()
    changed_key[32:].normal_()
    changed_value[32:].normal_()
    changed_query[32:].normal_()
    changed_cache = torch.zeros_like(one_shot_cache)
    impl._prefill_staging.clear()
    _pack_prefill_at_boundaries(
        impl,
        changed_key,
        changed_value,
        changed_query,
        (prompt_length,),
        table,
        changed_cache,
    )
    assert torch.equal(one_shot_cache[:2], changed_cache[:2])


@requires_cuda
def test_direct_cached_prefix_attention_keeps_shared_pages_immutable() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    impl = _make_prefill_pack_impl()
    prompt_length, cached_prefix = 80, 32
    key = torch.randn(prompt_length, 2, 32, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    query = torch.randn(prompt_length, 8, 32, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(
        8,
        2,
        impl.block_layout.head_page_bytes,
        dtype=torch.uint8,
        device=device,
    )
    cold_table = torch.arange(5, dtype=torch.int32, device=device).unsqueeze(0)
    _pack_prefill_at_boundaries(
        impl, key, value, query, (prompt_length,), cold_table, cache
    )
    shared_ids = cold_table[0, : cached_prefix // impl.block_size]
    shared_before = cache.index_select(0, shared_ids).clone()

    replay_table = torch.tensor([[0, 1, 5, 6, 7]], dtype=torch.int32, device=device)
    impl._prefill_staging.clear()
    _pack_prefill_at_boundaries(
        impl,
        key,
        value,
        query,
        (prompt_length,),
        replay_table,
        cache,
        cached_prefix=cached_prefix,
    )
    assert torch.equal(cache.index_select(0, shared_ids), shared_before)
    assert torch.count_nonzero(cache[5:7]) > 0

    replay_query = query[cached_prefix:]
    result = triton_hyquant_prefix_attention(
        replay_query,
        cache,
        replay_table,
        torch.tensor([0, replay_query.shape[0]], dtype=torch.int32, device=device),
        torch.tensor([cached_prefix], dtype=torch.int32, device=device),
        impl.block_layout,
        impl.head_size**-0.5,
        max_query_len=replay_query.shape[0],
    )
    assert result is not None
    actual_output, actual_lse = result
    prefix_key, prefix_value = materialize_hyquant_blocks(
        cache, shared_ids, impl.block_layout
    )
    expected_output, expected_lse = hyquant_prefix_attention_reference(
        replay_query,
        prefix_key,
        prefix_value,
        impl.head_size**-0.5,
    )
    assert get_hyquant_prefix_status() == "triton_tiled"
    assert torch.allclose(actual_output, expected_output, atol=4e-3, rtol=3e-2)
    assert torch.allclose(actual_lse, expected_lse, atol=2e-5, rtol=2e-5)
    assert torch.equal(cache.index_select(0, shared_ids), shared_before)


@requires_cuda
@pytest.mark.parametrize(
    ("query_heads", "kv_heads", "ratio"),
    [(4, 4, 0.0), (8, 2, 1 / 16), (8, 1, 1.0), (8, 4, 0.25)],
)
def test_direct_prefix_attention_matches_multi_request_reference(
    query_heads: int,
    kv_heads: int,
    ratio: float,
) -> None:
    """Direct pages support MHA/GQA/MQA, zero hits, and request offsets."""
    torch.manual_seed(31)
    device = torch.device("cuda")
    block_size, head_size = 16, 32
    layout = HyQuantBlockLayout.from_params(block_size, head_size, 16, ratio)
    source_key = torch.randn(
        5,
        block_size,
        kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    source_value = torch.randn_like(source_key)
    reference_queries = torch.randn(
        5, query_heads, head_size, dtype=torch.bfloat16, device=device
    )
    anchors = select_important_token_indices(
        source_key.reshape(-1, kv_heads, head_size),
        reference_queries,
        block_size,
        ratio,
    )
    cache = torch.zeros(
        8,
        kv_heads,
        layout.head_page_bytes,
        dtype=torch.uint8,
        device=device,
    )
    physical_ids = torch.tensor([7, 2, 5, 1, 6], dtype=torch.int32, device=device)
    pack_hyquant_blocks(
        source_key,
        source_value,
        physical_ids,
        anchors,
        cache,
        layout,
    )
    block_table = torch.tensor(
        [[7, 2, 0], [0, 0, 0], [5, 1, 6]], dtype=torch.int32, device=device
    )
    query_lens = (3, 2, 4)
    query = torch.randn(
        sum(query_lens),
        query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    query_start_loc = torch.tensor(
        [0, query_lens[0], sum(query_lens[:2]), sum(query_lens)],
        dtype=torch.int32,
        device=device,
    )
    prefix_lens = torch.tensor([32, 0, 48], dtype=torch.int32, device=device)
    before = cache.clone()

    result = triton_hyquant_prefix_attention(
        query,
        cache,
        block_table,
        query_start_loc,
        prefix_lens,
        layout,
        head_size**-0.5,
        max_query_len=max(query_lens),
    )
    assert result is not None
    actual_output, actual_lse = result

    query_start = 0
    for request_index, query_len in enumerate(query_lens):
        query_end = query_start + query_len
        prefix_len = int(prefix_lens[request_index])
        if prefix_len == 0:
            assert torch.count_nonzero(actual_output[query_start:query_end]) == 0
            assert torch.isneginf(actual_lse[:, query_start:query_end]).all()
        else:
            block_count = prefix_len // block_size
            dense_key, dense_value = materialize_hyquant_blocks(
                cache,
                block_table[request_index, :block_count],
                layout,
            )
            expected_output, expected_lse = hyquant_prefix_attention_reference(
                query[query_start:query_end],
                dense_key,
                dense_value,
                head_size**-0.5,
            )
            assert torch.allclose(
                actual_output[query_start:query_end],
                expected_output,
                atol=4e-3,
                rtol=3e-2,
            )
            assert torch.allclose(
                actual_lse[:, query_start:query_end],
                expected_lse,
                atol=2e-5,
                rtol=2e-5,
            )
        query_start = query_end
    assert get_hyquant_prefix_status() == "triton_tiled"
    assert torch.equal(cache, before)


@requires_cuda
def test_groupdot_prefix_and_fa2_suffix_match_dense_prefill() -> None:
    """The production hit path merges compact prefix and BF16 suffix states."""
    torch.manual_seed(37)
    device = torch.device("cuda")
    block_size, head_size, query_heads, kv_heads = 16, 128, 8, 2
    layout = HyQuantBlockLayout.from_params(block_size, head_size, 32, 1 / 16)
    prefix_key = torch.randn(
        2,
        block_size,
        kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    prefix_value = torch.randn_like(prefix_key)
    anchor_query = torch.randn(
        2, query_heads, head_size, dtype=torch.bfloat16, device=device
    )
    anchors = select_important_token_indices(
        prefix_key.reshape(-1, kv_heads, head_size),
        anchor_query,
        block_size,
        1 / 16,
    )
    cache = torch.zeros(
        4,
        kv_heads,
        layout.head_page_bytes,
        dtype=torch.uint8,
        device=device,
    )
    physical_ids = torch.tensor([3, 1], dtype=torch.int32, device=device)
    pack_hyquant_blocks(
        prefix_key,
        prefix_value,
        physical_ids,
        anchors,
        cache,
        layout,
    )
    block_table = torch.tensor([[3, 1], [0, 0]], dtype=torch.int32, device=device)
    suffix_lens = (5, 3)
    query_start_loc_cpu = [0, suffix_lens[0], sum(suffix_lens)]
    query_start_loc = torch.tensor(
        query_start_loc_cpu, dtype=torch.int32, device=device
    )
    query = torch.randn(
        sum(suffix_lens),
        query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    suffix_keys = [
        torch.randn(length, kv_heads, head_size, dtype=torch.bfloat16, device=device)
        for length in suffix_lens
    ]
    suffix_values = [torch.randn_like(item) for item in suffix_keys]
    cached_prefix_lens = torch.tensor([32, 0], dtype=torch.int32, device=device)
    output = torch.empty_like(query)
    impl = object.__new__(HyQuantAttentionImpl)
    impl.block_layout = layout
    impl.scale = head_size**-0.5
    impl.fa_version = 2
    before = cache.clone()

    assert impl._run_mixed_prefix_prefill(
        query,
        cache,
        suffix_keys,
        suffix_values,
        output,
        query_start_loc,
        query_start_loc_cpu,
        query_start_loc,
        query_start_loc_cpu,
        block_table,
        cached_prefix_lens,
        max(suffix_lens),
        max(suffix_lens),
    )

    restored_key, restored_value = materialize_hyquant_blocks(
        cache, physical_ids, layout
    )
    full_keys = [
        torch.cat((restored_key, suffix_keys[0])),
        suffix_keys[1],
    ]
    full_values = [
        torch.cat((restored_value, suffix_values[0])),
        suffix_values[1],
    ]
    expected = torch.empty_like(output)
    impl._run_dense_prefill_fallback(
        query,
        torch.cat(full_keys),
        torch.cat(full_values),
        expected,
        query_start_loc_cpu,
        [0, full_keys[0].shape[0], sum(item.shape[0] for item in full_keys)],
    )
    assert get_hyquant_prefix_status() == "triton_groupdot"
    assert torch.allclose(output, expected, atol=5e-3, rtol=4e-2)
    assert torch.equal(cache, before)


@requires_cuda
@pytest.mark.parametrize("query_heads,kv_heads", [(8, 4), (8, 1), (4, 4)])
def test_mixed_decode_matches_reference(query_heads: int, kv_heads: int) -> None:
    device = torch.device("cuda")
    block_size, head_size = 16, 128
    num_blocks, sequence_length = 32, 512
    layout = HyQuantBlockLayout.from_params(block_size, head_size, 32, 1 / 16)
    key = torch.randn(
        num_blocks, block_size, kv_heads, head_size, device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    source_query = torch.randn(query_heads, head_size, device=device)
    anchors = select_important_token_indices(
        key.reshape(-1, kv_heads, head_size), source_query, block_size, 1 / 16
    )
    cache = torch.zeros(
        num_blocks, kv_heads, layout.head_page_bytes, dtype=torch.uint8, device=device
    )
    block_ids = torch.arange(num_blocks, device=device, dtype=torch.int64)
    pack_hyquant_blocks(key, value, block_ids, anchors, cache, layout)
    query = torch.randn(1, query_heads, head_size, device=device, dtype=torch.bfloat16)
    hot_key = key.reshape(-1, kv_heads, head_size).permute(1, 0, 2).unsqueeze(0)
    hot_value = value.reshape(-1, kv_heads, head_size).permute(1, 0, 2).unsqueeze(0)
    block_table = block_ids[None].to(torch.int32)
    seq_lens = torch.tensor([sequence_length], device=device, dtype=torch.int32)
    prompt_lens = seq_lens.clone()
    state_slots = torch.zeros(1, device=device, dtype=torch.int32)
    actual = triton_hyquant_block_decode_attention(
        query,
        cache,
        hot_key,
        hot_value,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        layout,
        256,
        64,
        head_size**-0.5,
    )
    assert actual is not None
    reference = hyquant_block_decode_attention(
        query,
        cache,
        hot_key,
        hot_value,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        layout,
        256,
        64,
        head_size**-0.5,
    )
    assert torch.allclose(actual, reference, atol=3e-3, rtol=3e-2)


@requires_cuda
@pytest.mark.parametrize("query_heads,kv_heads", [(8, 4), (8, 1), (4, 4)])
def test_fused_current_decode_matches_explicit_store(
    query_heads: int, kv_heads: int
) -> None:
    """The fused append must match explicit hot-store attention and state."""
    device = torch.device("cuda")
    block_size, head_size = 16, 128
    # Keep one current row outside the ring for the fused path.  The previous
    # 256 rows are already in the hot window and the compact prefix is stable.
    sequence_length, prompt_length, committed = 513, 512, 256
    layout = HyQuantBlockLayout.from_params(block_size, head_size, 32, 1 / 16)
    num_blocks = (sequence_length + block_size - 1) // block_size
    history = torch.randn(
        committed,
        kv_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    history_v = torch.randn_like(history)
    current_k = torch.randn(1, kv_heads, head_size, device=device, dtype=torch.bfloat16)
    current_v = torch.randn_like(current_k)
    query = torch.randn(1, query_heads, head_size, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
    anchors = select_important_token_indices(history, query[0], block_size, 1 / 16)
    base_cache = torch.zeros(
        num_blocks, kv_heads, layout.head_page_bytes, dtype=torch.uint8, device=device
    )
    pack_hyquant_blocks(
        history.view(-1, block_size, kv_heads, head_size),
        history_v.view(-1, block_size, kv_heads, head_size),
        block_table[0, : committed // block_size],
        anchors,
        base_cache,
        layout,
    )
    hot_capacity = get_hyquant_hot_capacity(256, 64, block_size)
    base_hot_k = torch.zeros(
        1, kv_heads, hot_capacity, head_size, dtype=torch.bfloat16, device=device
    )
    base_hot_v = torch.zeros_like(base_hot_k)
    old_positions = torch.arange(committed, sequence_length - 1, device=device)
    base_hot_k[0, :, old_positions.remainder(hot_capacity)] = history.permute(1, 0, 2)
    base_hot_v[0, :, old_positions.remainder(hot_capacity)] = history_v.permute(1, 0, 2)
    seq_lens = torch.tensor([sequence_length], device=device, dtype=torch.int32)
    prompt_lens = torch.tensor([prompt_length], device=device, dtype=torch.int32)
    state_slots = torch.zeros(1, device=device, dtype=torch.int32)
    current_slot = torch.tensor([sequence_length - 1], device=device, dtype=torch.int64)
    current_position = torch.tensor(
        [sequence_length - 1], device=device, dtype=torch.int64
    )
    token_to_req = torch.zeros(1, device=device, dtype=torch.int32)

    explicit_cache = base_cache.clone()
    explicit_hot_k = base_hot_k.clone()
    explicit_hot_v = base_hot_v.clone()
    assert triton_hyquant_store_hot(
        current_k,
        current_v,
        explicit_hot_k,
        explicit_hot_v,
        current_slot,
        current_position,
        token_to_req,
        state_slots,
        seq_lens,
        prompt_lens,
        block_size,
        256,
        64,
    )
    explicit = triton_hyquant_block_decode_attention(
        query,
        explicit_cache,
        explicit_hot_k,
        explicit_hot_v,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        layout,
        256,
        64,
        head_size**-0.5,
        mid_o_buf=torch.empty(1, query_heads, 32, 129, device=device),
        max_seq_len=sequence_length,
    )
    fused_cache = base_cache.clone()
    fused_hot_k = base_hot_k.clone()
    fused_hot_v = base_hot_v.clone()
    fused = triton_hyquant_block_decode_attention(
        query,
        fused_cache,
        fused_hot_k,
        fused_hot_v,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        layout,
        256,
        64,
        head_size**-0.5,
        mid_o_buf=torch.empty(1, query_heads, 32, 129, device=device),
        max_seq_len=sequence_length,
        current_key=current_k,
        current_value=current_v,
        fuse_current_store=True,
    )
    assert explicit is not None and fused is not None
    assert torch.allclose(fused, explicit, atol=4e-3, rtol=4e-2)
    current_ring = (sequence_length - 1) % hot_capacity
    assert torch.equal(
        fused_hot_k[0, :, current_ring], explicit_hot_k[0, :, current_ring]
    )
    assert torch.equal(
        fused_hot_v[0, :, current_ring], explicit_hot_v[0, :, current_ring]
    )


@requires_cuda
@pytest.mark.parametrize(
    ("decode_path", "expected_status"),
    [
        ("int8", "triton_hyquant_groupdot_splitk"),
        ("groupdot", "triton_hyquant_groupdot_splitk"),
        ("grouped", "triton_hyquant_grouped_splitk"),
    ],
)
def test_fused_multi_request_decode_supports_distinct_current_kv_strides(
    monkeypatch: pytest.MonkeyPatch,
    decode_path: str,
    expected_status: str,
) -> None:
    """Fused grouped decode must use independent K/V batch strides.

    Fused QKV projections commonly produce a contiguous K view and a V view
    whose token stride still includes the unused Q/K columns.  A shared stride
    makes request zero appear correct while reading padding for later requests.
    Keep the request state slots and physical pages nontrivial so both the
    current-row load and the split-K reduction are exercised.
    """
    monkeypatch.setenv(
        "VLLM_HYQUANT_DECODE_INT8_QK", "1" if decode_path == "int8" else "0"
    )
    if decode_path == "grouped":
        monkeypatch.setenv("VLLM_HYQUANT_DISABLE_GROUPDOT", "1")
    else:
        monkeypatch.delenv("VLLM_HYQUANT_DISABLE_GROUPDOT", raising=False)
    monkeypatch.delenv("VLLM_HYQUANT_DISABLE_GROUPED_DECODE", raising=False)
    monkeypatch.setenv("VLLM_HYQUANT_GROUPED_STRICT", "1")

    device = torch.device("cuda")
    batch_size, query_heads, kv_heads, head_size = 2, 8, 2, 128
    block_size = 16
    sequence_length, prompt_length, committed = 513, 512, 256
    layout = HyQuantBlockLayout.from_params(block_size, head_size, 32, 1 / 16)
    num_logical_blocks = (sequence_length + block_size - 1) // block_size
    num_history_blocks = committed // block_size

    torch.manual_seed(41)
    history = torch.randn(
        batch_size,
        committed,
        kv_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    history_v = torch.randn_like(history)
    query = torch.randn(
        batch_size,
        query_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )

    # Give every request a disjoint page-table range and swap state slots.  The
    # decode kernels must use the request row, rather than assuming slot == row.
    pages_per_request = num_logical_blocks
    total_pages = batch_size * pages_per_request
    block_table = (
        torch.arange(total_pages, device=device, dtype=torch.int32)
        .view(batch_size, pages_per_request)
        .flip(1)
    )
    anchors = torch.cat(
        [
            select_important_token_indices(
                history[index], query[index], block_size, 1 / 16
            )
            for index in range(batch_size)
        ],
        dim=0,
    )
    cache = torch.zeros(
        total_pages,
        kv_heads,
        layout.head_page_bytes,
        dtype=torch.uint8,
        device=device,
    )
    pack_hyquant_blocks(
        history.reshape(
            batch_size * num_history_blocks, block_size, kv_heads, head_size
        ),
        history_v.reshape(
            batch_size * num_history_blocks, block_size, kv_heads, head_size
        ),
        block_table[:, :num_history_blocks].reshape(-1),
        anchors,
        cache,
        layout,
    )

    # Deliberately reverse the physical hot-ring slots relative to requests.
    state_slots = torch.tensor([1, 0], dtype=torch.int32, device=device)
    hot_capacity = get_hyquant_hot_capacity(256, 64, block_size)
    hot_key = torch.zeros(
        batch_size,
        kv_heads,
        hot_capacity,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    hot_value = torch.zeros_like(hot_key)
    old_positions = torch.arange(committed, sequence_length - 1, device=device)
    history_hot_k = history.permute(0, 2, 1, 3)
    history_hot_v = history_v.permute(0, 2, 1, 3)
    ring_positions = old_positions.remainder(hot_capacity)
    for request_index, state_slot in enumerate((1, 0)):
        hot_key[state_slot, :, ring_positions] = history_hot_k[request_index]
        hot_value[state_slot, :, ring_positions] = history_hot_v[request_index]

    seq_lens = torch.full(
        (batch_size,), sequence_length, dtype=torch.int32, device=device
    )
    prompt_lens = torch.full(
        (batch_size,), prompt_length, dtype=torch.int32, device=device
    )
    current_key = torch.randn(
        batch_size,
        kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    value_token_stride = kv_heads * head_size * 3
    value_storage_size = (
        (batch_size - 1) * value_token_stride + (kv_heads - 1) * head_size + head_size
    )
    value_storage = torch.full(
        (value_storage_size,),
        17.0,
        dtype=torch.bfloat16,
        device=device,
    )
    current_value = torch.as_strided(
        value_storage,
        (batch_size, kv_heads, head_size),
        (value_token_stride, head_size, 1),
    )
    current_value.copy_(torch.randn_like(current_key))
    assert current_key.stride(0) != current_value.stride(0)
    assert current_key.stride(1) == current_value.stride(1)

    explicit_hot_k = hot_key.clone()
    explicit_hot_v = hot_value.clone()
    positions = torch.full(
        (batch_size,), sequence_length - 1, dtype=torch.int64, device=device
    )
    slot_mapping = positions.clone()
    token_to_req = torch.arange(batch_size, dtype=torch.int32, device=device)
    assert triton_hyquant_store_hot(
        current_key,
        current_value,
        explicit_hot_k,
        explicit_hot_v,
        slot_mapping,
        positions,
        token_to_req,
        state_slots,
        seq_lens,
        prompt_lens,
        block_size,
        256,
        64,
    )
    workspace_shape = (batch_size, query_heads, 4, 129)
    explicit = triton_hyquant_block_decode_attention(
        query,
        cache.clone(),
        explicit_hot_k,
        explicit_hot_v,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        layout,
        256,
        64,
        head_size**-0.5,
        mid_o_buf=torch.empty(workspace_shape, dtype=torch.float32, device=device),
        max_seq_len=sequence_length,
    )
    fused_hot_k = hot_key.clone()
    fused_hot_v = hot_value.clone()
    fused = triton_hyquant_block_decode_attention(
        query,
        cache.clone(),
        fused_hot_k,
        fused_hot_v,
        block_table,
        seq_lens,
        prompt_lens,
        state_slots,
        layout,
        256,
        64,
        head_size**-0.5,
        mid_o_buf=torch.empty(workspace_shape, dtype=torch.float32, device=device),
        max_seq_len=sequence_length,
        current_key=current_key,
        current_value=current_value,
        fuse_current_store=True,
    )
    assert explicit is not None and fused is not None
    assert get_hyquant_block_decode_status() == expected_status
    assert torch.allclose(fused, explicit, atol=4e-3, rtol=4e-2)
    current_ring = (sequence_length - 1) % hot_capacity
    assert torch.equal(
        fused_hot_k[:, :, current_ring], explicit_hot_k[:, :, current_ring]
    )
    assert torch.equal(
        fused_hot_v[:, :, current_ring], explicit_hot_v[:, :, current_ring]
    )
