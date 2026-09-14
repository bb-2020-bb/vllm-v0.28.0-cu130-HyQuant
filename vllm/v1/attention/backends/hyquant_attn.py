"""vLLM V1 backend for periodic-window HyQuant KV caching."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.hyquant_block import (
    hyquant_block_decode_attention,
    pack_hyquant_blocks,
    select_important_token_indices,
    triton_hyquant_block_decode_attention,
    warmup_hyquant_decode_kernel,
)
from vllm.v1.attention.ops.hyquant_common import (
    HYQUANT_CACHE_DTYPE,
    HyQuantBlockLayout,
    get_hyquant_committed_len,
    get_hyquant_group_size,
    get_hyquant_hot_capacity,
    get_hyquant_new_completed_block_range,
    get_hyquant_top_ratio,
    get_hyquant_window_config,
)
from vllm.v1.attention.ops.hyquant_decode_groupdot import (
    get_hyquant_decode_launch_config,
    get_hyquant_decode_num_stages,
    get_hyquant_decode_split_count,
)
from vllm.v1.attention.ops.hyquant_decode_int8 import (
    hyquant_int8qk_decode,
    warmup_hyquant_int8qk_kernel,
)
from vllm.v1.attention.ops.hyquant_hot import (
    triton_hyquant_store_hot,
    warmup_hyquant_hot_kernels,
)
from vllm.v1.attention.ops.hyquant_prefill import (
    triton_hyquant_prefix_attention,
    warmup_hyquant_prefix_kernel,
)
from vllm.v1.attention.ops.hyquant_retire import (
    triton_hyquant_retire_blocks,
    warmup_hyquant_retire_kernels,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)
_HYQUANT_DEBUG = os.environ.get("VLLM_HYQUANT_DEBUG", "0") == "1"
_HYQUANT_DEBUGGED_PHASES: set[str] = set()


def _hyquant_retirement_needed_for_rows(
    seq_lens_cpu: torch.Tensor,
    prompt_lens_cpu: torch.Tensor,
    num_decodes: int,
    block_size: int,
    window_size: int,
    retire_interval: int,
) -> bool:
    """Check retirement boundaries once while building step metadata.

    The previous implementation recomputed this condition in every attention
    layer.  Decode metadata is shared by all layers, so doing the small CPU
    check here lets ordinary (non-boundary) steps avoid the Python/Torch
    retirement path entirely.
    """
    for request_index in range(num_decodes):
        sequence_length = int(seq_lens_cpu[request_index])
        prompt_length = int(prompt_lens_cpu[request_index])
        if sequence_length <= prompt_length:
            continue
        previous_length = max(sequence_length - 1, prompt_length)
        current_committed = get_hyquant_committed_len(
            sequence_length,
            prompt_length,
            block_size,
            window_size,
            retire_interval,
        )
        previous_committed = get_hyquant_committed_len(
            previous_length,
            prompt_length,
            block_size,
            window_size,
            retire_interval,
        )
        if current_committed > previous_committed:
            return True
    return False


@dataclass
class HyQuantMetadata(AttentionMetadata):
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    prefill_query_start_loc: torch.Tensor
    prefill_kv_start_loc: torch.Tensor
    prefill_kv_start_loc_cpu: torch.Tensor
    prefill_suffix_kv_start_loc: torch.Tensor
    prefill_suffix_kv_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    prompt_lens: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    cached_prefix_lens: torch.Tensor
    cached_prefix_lens_cpu: torch.Tensor
    state_slots: torch.Tensor
    state_slots_cpu: torch.Tensor
    token_to_req: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    max_query_len: int
    max_prefill_query_len: int
    max_prefill_seq_len: int
    max_seq_len: int
    max_decode_seq_len: int
    num_decodes: int
    num_decode_tokens: int
    mid_o_buf: torch.Tensor | None = None
    retire_fast_path: bool = True
    retirement_needed: bool = False
    # Decode launch decisions are step-dependent (batch/context length), but
    # identical for every layer in this attention group.  Resolve them while
    # building shared metadata instead of re-reading environment variables and
    # running the policy once per layer.
    decode_num_splits: int = 1
    decode_block_kv: int = 128
    decode_block_h: int = 4
    decode_num_warps: int = 4
    decode_num_stages: int = 2
    decode_use_simt: bool = False
    decode_coalesce_block_table: bool = True
    decode_coalesce_scales: bool = True


@dataclass
class _HyQuantPrefillState:
    key: torch.Tensor
    value: torch.Tensor
    filled_len: int
    prompt_len: int
    cached_prefix_len: int


class HyQuantMetadataBuilder(AttentionMetadataBuilder[HyQuantMetadata]):
    # Pure one-token decode is graph-safe between retirement boundaries: the
    # grouped kernel updates the BF16 hot ring in-place and all metadata/cache
    # pointers are persistent buffers.  The runner may still select eager for
    # a boundary step (where host-side page retirement is required).
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self._num_query_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self._token_to_req = torch.zeros(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )
        max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self._prefill_kv_start_loc = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self._prefill_kv_start_loc_cpu = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )
        self._prefill_suffix_kv_start_loc = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self._prefill_suffix_kv_start_loc_cpu = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> HyQuantMetadata:
        del common_prefix_len, fast_build
        if common_attn_metadata.hyquant_state_slots is None:
            raise RuntimeError("HyQuant state-slot metadata was not initialized")
        if common_attn_metadata.hyquant_prompt_lens is None:
            raise RuntimeError("HyQuant prompt-length metadata was not initialized")
        if common_attn_metadata.hyquant_cached_prefix_lens is None:
            raise RuntimeError("HyQuant cached-prefix metadata was not initialized")
        if common_attn_metadata.positions is None:
            raise RuntimeError("HyQuant requires logical token positions")
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if seq_lens_cpu is None:
            seq_lens_cpu = common_attn_metadata.seq_lens.detach().cpu()
        prompt_lens_cpu = common_attn_metadata.hyquant_prompt_lens_cpu
        if prompt_lens_cpu is None:
            prompt_lens_cpu = common_attn_metadata.hyquant_prompt_lens.detach().cpu()
        state_slots_cpu = common_attn_metadata.hyquant_state_slots_cpu
        if state_slots_cpu is None:
            state_slots_cpu = common_attn_metadata.hyquant_state_slots.detach().cpu()
        cached_prefix_lens_cpu = common_attn_metadata.hyquant_cached_prefix_lens_cpu
        if cached_prefix_lens_cpu is None:
            cached_prefix_lens_cpu = (
                common_attn_metadata.hyquant_cached_prefix_lens.detach().cpu()
            )
        num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=1,
            treat_short_extends_as_decodes=False,
        )
        max_decode_seq_len = int(seq_lens_cpu[:num_decodes].max()) if num_decodes else 0
        prefill_starts = (
            common_attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
        )
        query_lens_cpu = (
            common_attn_metadata.query_start_loc_cpu[1:]
            - common_attn_metadata.query_start_loc_cpu[:-1]
        )
        max_prefill_query_len = (
            int(query_lens_cpu[num_decodes:].max())
            if num_decodes < common_attn_metadata.num_reqs
            else 0
        )
        num_prefills = common_attn_metadata.num_reqs - num_decodes
        prefill_kv_start_loc_cpu = self._prefill_kv_start_loc_cpu[: num_prefills + 1]
        prefill_kv_start_loc_cpu.zero_()
        prefill_suffix_kv_start_loc_cpu = self._prefill_suffix_kv_start_loc_cpu[
            : num_prefills + 1
        ]
        prefill_suffix_kv_start_loc_cpu.zero_()
        if num_prefills:
            prefill_seq_lens = seq_lens_cpu[
                num_decodes : common_attn_metadata.num_reqs
            ].to(dtype=torch.int32)
            prefill_cached_lens = cached_prefix_lens_cpu[
                num_decodes : common_attn_metadata.num_reqs
            ].to(dtype=torch.int32)
            torch.cumsum(
                prefill_seq_lens,
                dim=0,
                out=prefill_kv_start_loc_cpu[1:],
            )
            torch.cumsum(
                prefill_seq_lens - prefill_cached_lens,
                dim=0,
                out=prefill_suffix_kv_start_loc_cpu[1:],
            )
            self._prefill_kv_start_loc[: num_prefills + 1].copy_(
                prefill_kv_start_loc_cpu
            )
            self._prefill_suffix_kv_start_loc[: num_prefills + 1].copy_(
                prefill_suffix_kv_start_loc_cpu
            )
        max_prefill_seq_len = (
            int(seq_lens_cpu[num_decodes : common_attn_metadata.num_reqs].max())
            if num_prefills
            else 0
        )
        retire_fast_path = common_attn_metadata._seq_lens_cpu is not None
        retirement_needed = False
        if num_decode_tokens:
            retirement_needed = _hyquant_retirement_needed_for_rows(
                seq_lens_cpu,
                prompt_lens_cpu,
                num_decodes,
                int(self.kv_cache_spec.block_size),
                *get_hyquant_window_config(),
            )

        # The grouped split-K kernel uses a shared workspace.  Resolve its
        # view once per metadata object (shared by all layers) instead of
        # asking the workspace manager from every layer's forward() call.
        mid_o_buf = None
        decode_group_size = (
            self._num_query_heads // self.kv_cache_spec.num_kv_heads
            if self.kv_cache_spec.num_kv_heads
            else self._num_query_heads
        )
        if num_decode_tokens > 0:
            decode_num_splits = get_hyquant_decode_split_count(
                num_decode_tokens,
                max_decode_seq_len,
                decode_group_size,
            )
            (
                decode_block_kv,
                decode_block_h,
                decode_num_warps,
            ) = get_hyquant_decode_launch_config(
                num_decode_tokens,
                max_decode_seq_len,
                decode_group_size,
            )
            decode_num_stages = get_hyquant_decode_num_stages(
                num_decode_tokens, max_decode_seq_len
            )
        else:
            decode_num_splits = 1
            decode_block_kv, decode_block_h, decode_num_warps = 128, 4, 4
            decode_num_stages = 2
        needs_decode_workspace = decode_num_splits > 1
        decode_use_simt = os.environ.get("VLLM_HYQUANT_DECODE_SIMT", "0") == "1"
        decode_coalesce_block_table = (
            os.environ.get("VLLM_HYQUANT_DECODE_COALESCE_BLOCK_TABLE", "1") != "0"
        )
        decode_coalesce_scales = (
            os.environ.get("VLLM_HYQUANT_DECODE_COALESCE_SCALES", "1") != "0"
        )
        if needs_decode_workspace and is_workspace_manager_initialized():
            block_d = 1 << (self.kv_cache_spec.head_size - 1).bit_length()
            try:
                (mid_o_buf,) = current_workspace_manager().get_simultaneous(
                    (
                        (
                            num_decode_tokens,
                            self._num_query_heads,
                            64,
                            block_d + 1,
                        ),
                        torch.float32,
                    )
                )
            except AssertionError:
                # A locked manager may intentionally omit this specialization;
                # the launcher will report a capture-safe miss and use its
                # validated fallback.
                mid_o_buf = None
        return HyQuantMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            prefill_query_start_loc=prefill_starts,
            prefill_kv_start_loc=self._prefill_kv_start_loc[: num_prefills + 1],
            prefill_kv_start_loc_cpu=prefill_kv_start_loc_cpu,
            prefill_suffix_kv_start_loc=self._prefill_suffix_kv_start_loc[
                : num_prefills + 1
            ],
            prefill_suffix_kv_start_loc_cpu=prefill_suffix_kv_start_loc_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            prompt_lens=common_attn_metadata.hyquant_prompt_lens,
            prompt_lens_cpu=prompt_lens_cpu,
            cached_prefix_lens=common_attn_metadata.hyquant_cached_prefix_lens,
            cached_prefix_lens_cpu=cached_prefix_lens_cpu,
            state_slots=common_attn_metadata.hyquant_state_slots,
            state_slots_cpu=state_slots_cpu,
            token_to_req=common_attn_metadata.token_to_req_indices(self._token_to_req),
            positions=common_attn_metadata.positions,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_prefill_query_len=max_prefill_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            mid_o_buf=mid_o_buf,
            retire_fast_path=retire_fast_path,
            retirement_needed=retirement_needed,
            decode_num_splits=decode_num_splits,
            decode_block_kv=decode_block_kv,
            decode_block_h=decode_block_h,
            decode_num_warps=decode_num_warps,
            decode_num_stages=decode_num_stages,
            decode_use_simt=decode_use_simt,
            decode_coalesce_block_table=decode_coalesce_block_table,
            decode_coalesce_scales=decode_coalesce_scales,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> HyQuantMetadata:
        metadata = self.build(0, common_attn_metadata)
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        metadata.max_seq_len = max_model_len
        metadata.max_decode_seq_len = max_model_len
        # A replay can carry any live sequence length up to max_model_len even
        # though the capture dummy rows are initialized with a short context.
        # Keep the conservative graph launch policy used by the previous
        # implementation and resolve it once here; all layers then share the
        # same decision through metadata.
        if metadata.num_decode_tokens > 0:
            decode_group_size = (
                self._num_query_heads // self.kv_cache_spec.num_kv_heads
                if self.kv_cache_spec.num_kv_heads
                else self._num_query_heads
            )
            metadata.decode_num_splits = get_hyquant_decode_split_count(
                metadata.num_decode_tokens,
                max_model_len,
                decode_group_size,
            )
            (
                metadata.decode_block_kv,
                metadata.decode_block_h,
                metadata.decode_num_warps,
            ) = get_hyquant_decode_launch_config(
                metadata.num_decode_tokens,
                max_model_len,
                decode_group_size,
            )
            metadata.decode_num_stages = get_hyquant_decode_num_stages(
                metadata.num_decode_tokens, max_model_len
            )
            if metadata.decode_num_splits > 1 and metadata.mid_o_buf is None:
                block_d = 1 << (self.kv_cache_spec.head_size - 1).bit_length()
                if is_workspace_manager_initialized():
                    try:
                        (metadata.mid_o_buf,) = (
                            current_workspace_manager().get_simultaneous(
                                (
                                    (
                                        metadata.num_decode_tokens,
                                        self._num_query_heads,
                                        64,
                                        block_d + 1,
                                    ),
                                    torch.float32,
                                )
                            )
                        )
                    except AssertionError:
                        metadata.mid_o_buf = None
        if metadata.seq_lens.numel():
            metadata.seq_lens.fill_(1)
        metadata.retire_fast_path = False
        metadata.retirement_needed = False
        return metadata

    def update_block_table(
        self,
        metadata: HyQuantMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> HyQuantMetadata:
        metadata.block_table = blk_table
        metadata.slot_mapping = slot_mapping
        return metadata


class HyQuantAttentionBackend(AttentionBackend):
    """Compact K4/V4 prefix, BF16 anchors, and a periodic BF16 window."""

    forward_includes_kv_cache_update: bool = True
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [HYQUANT_CACHE_DTYPE]

    @staticmethod
    def get_name() -> str:
        return "HYQUANT"

    @staticmethod
    def get_impl_cls() -> type[HyQuantAttentionImpl]:
        return HyQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[HyQuantMetadataBuilder]:
        return HyQuantMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 16 and head_size % 8 == 0

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == HYQUANT_CACHE_DTYPE

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major >= 8

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.kv_quant_mode != KVQuantMode.HYQUANT_K4V4:
            return spec
        layout = HyQuantBlockLayout.from_params(
            spec.block_size,
            spec.head_size,
            get_hyquant_group_size(),
            get_hyquant_top_ratio(),
        )
        return replace(
            spec,
            dtype=torch.uint8,
            state_content_bytes=layout.bytes_per_token,
            prefix_cache_private_tail_tokens=get_hyquant_window_config()[0],
        )

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = HYQUANT_CACHE_DTYPE,
    ) -> tuple[int, ...]:
        if cache_dtype_str != HYQUANT_CACHE_DTYPE:
            raise ValueError(f"HyQuant requires {HYQUANT_CACHE_DTYPE}")
        layout = HyQuantBlockLayout.from_params(
            block_size,
            head_size,
            get_hyquant_group_size(),
            get_hyquant_top_ratio(),
        )
        return num_blocks, num_kv_heads, layout.head_page_bytes

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return 1, 2, 0, 3
        return 0, 1, 2


class HyQuantAttentionImpl(AttentionImpl[HyQuantMetadata]):
    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = HYQUANT_CACHE_DTYPE,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        del kwargs, kv_sharing_target_layer_name
        if attn_type != AttentionType.DECODER:
            raise ValueError("HyQuant supports decoder self-attention only")
        if sliding_window is not None:
            raise ValueError("HyQuant does not support model sliding windows")
        if logits_soft_cap is not None:
            raise ValueError("HyQuant does not support logits soft caps")
        if alibi_slopes is not None:
            raise ValueError("HyQuant does not yet support ALiBi")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if num_heads % self.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        self.head_size = head_size
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        self.top_ratio = get_hyquant_top_ratio()
        self.group_size = get_hyquant_group_size()
        self.window_size, self.retire_interval = get_hyquant_window_config()
        self._fuse_current_store_enabled = (
            os.environ.get("VLLM_HYQUANT_FUSE_CURRENT_STORE", "1") != "0"
        )
        self._gpu_retire_enabled = os.environ.get("VLLM_HYQUANT_GPU_RETIRE", "1") != "0"
        # The retirement packer is device-metadata driven and can be captured
        # in the decode graph.  Keeping this policy explicit gives deployments
        # with an older graph/runtime a controlled fallback to the eager
        # boundary path.
        self._graph_retire_enabled = (
            self._gpu_retire_enabled
            and os.environ.get("VLLM_HYQUANT_GRAPH_RETIRE", "1") != "0"
        )
        self._every_step_retirement_enabled = (
            self._gpu_retire_enabled
            and os.environ.get("VLLM_HYQUANT_GPU_RETIRE_EVERY_STEP", "0") == "1"
        )
        self._grouped_strict = os.environ.get("VLLM_HYQUANT_GROUPED_STRICT", "0") == "1"

        vllm_config = get_current_vllm_config()
        block_size = int(vllm_config.cache_config.block_size)
        if self.window_size % block_size or self.retire_interval % block_size:
            raise ValueError("HyQuant window and retirement interval must align")
        self.block_size = block_size
        self.block_layout = HyQuantBlockLayout.from_params(
            block_size, head_size, self.group_size, self.top_ratio
        )
        device = vllm_config.device_config.device
        warmup_hyquant_prefix_kernel(
            self.block_layout,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            device,
        )
        self._int8_decode_enabled = (
            os.environ.get("VLLM_HYQUANT_DECODE_INT8_QK", "1") != "0"
            and self.block_size == 16
            and head_size == 128
            and self.group_size == 32
            and self.block_layout.anchor_count == 1
            and self.block_layout.num_groups == 4
            and self.num_heads // self.num_kv_heads == 4
        )
        self.hot_capacity = get_hyquant_hot_capacity(
            self.window_size, self.retire_interval, block_size
        )
        max_state_slots = int(vllm_config.scheduler_config.max_num_seqs)
        self.hot_key = torch.empty(
            max_state_slots,
            self.num_kv_heads,
            self.hot_capacity,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.hot_value = torch.empty_like(self.hot_key)
        # Chunked prefill uses BF16 FlashAttention over the complete prefix.
        # Keep the temporary prompt K/V by stable request slot until the final
        # chunk is packed into compact HyQuant pages, then release it.
        self._prefill_staging: dict[int, _HyQuantPrefillState] = {}
        # Anchor indices are a small persistent scratch buffer.  Indexing by
        # request state slot keeps it valid across scheduler reordering and
        # makes both retirement launches CUDA-graph capturable.
        self._retire_block_table_width = max(
            1,
            (int(vllm_config.model_config.max_model_len) + block_size - 1)
            // block_size,
        )
        self._retire_anchor_workspace = torch.empty(
            max_state_slots,
            self._retire_block_table_width,
            max(1, self.block_layout.anchor_count),
            dtype=torch.int16,
            device=device,
        )
        # The grouped decoder may select a 64-way split for a single long
        # request.  Reserve the same upper bound here so serving never falls
        # back to a per-layer temporary allocation when that specialization is
        # selected.
        self._max_decode_splits = 64
        self._decode_workspace_shape = (
            max_state_slots,
            self.num_heads,
            self._max_decode_splits,
            1 << (head_size - 1).bit_length(),
        )
        # The partial softmax buffer is shared by all attention layers through
        # vLLM's workspace manager.  Reserve it while the manager is mutable;
        # the forward path below also handles configurations where the manager
        # is initialized later during worker setup.
        if is_workspace_manager_initialized():
            manager = current_workspace_manager()
            if not manager.is_locked():
                manager.get_simultaneous(
                    (
                        (
                            self._decode_workspace_shape[0],
                            self._decode_workspace_shape[1],
                            self._decode_workspace_shape[2],
                            self._decode_workspace_shape[3] + 1,
                        ),
                        torch.float32,
                    )
                )
        self.fa_version = get_flash_attn_version(head_size=head_size)
        # Compile the two hot-store variants and the decode specialization once
        # during model setup. Generic vLLM warmup may miss one of these
        # constexpr combinations, which otherwise causes a first-request
        # latency spike.
        warmup_hyquant_hot_kernels(
            self.num_kv_heads,
            head_size,
            block_size,
            self.window_size,
            self.retire_interval,
            self.hot_capacity,
            device,
        )
        warmup_hyquant_decode_kernel(
            self.block_layout,
            self.num_kv_heads,
            self.num_heads,
            self.window_size,
            self.retire_interval,
            self.hot_capacity,
            self.scale,
            device,
        )
        if self._int8_decode_enabled:
            warmup_hyquant_int8qk_kernel(
                self.block_layout,
                self.num_kv_heads,
                self.num_heads,
                self.window_size,
                self.retire_interval,
                self.hot_capacity,
                self.scale,
                device,
                max_seq_len=int(vllm_config.model_config.max_model_len),
                max_state_slots=max_state_slots,
                block_table_width=(
                    int(vllm_config.model_config.max_model_len) + block_size - 1
                )
                // block_size,
                fuse_current_store=self._fuse_current_store_enabled,
            )
        if self._gpu_retire_enabled:
            warmup_hyquant_retire_kernels(
                self.block_layout,
                self.num_kv_heads,
                self.num_heads,
                self.window_size,
                self.retire_interval,
                self.hot_capacity,
                device,
                max_blocks=self._retire_block_table_width,
            )

    def _run_dense_prefill_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_start_loc_cpu: list[int],
        key_start_loc_cpu: list[int] | None = None,
    ) -> torch.Tensor:
        """Reference rectangular causal prefill for environments without FA2."""
        import torch.nn.functional as F

        if key_start_loc_cpu is None:
            key_start_loc_cpu = query_start_loc_cpu
        if len(key_start_loc_cpu) != len(query_start_loc_cpu):
            raise ValueError("HyQuant prefill Q/KV request counts do not match")
        for request_index in range(len(query_start_loc_cpu) - 1):
            q_start = query_start_loc_cpu[request_index]
            q_end = query_start_loc_cpu[request_index + 1]
            k_start = key_start_loc_cpu[request_index]
            k_end = key_start_loc_cpu[request_index + 1]
            if q_end <= q_start:
                continue
            query_len = q_end - q_start
            key_len = k_end - k_start
            context_len = key_len - query_len
            if context_len < 0:
                raise ValueError("HyQuant prefill has fewer KV rows than query rows")
            # SDPA expects [batch, heads, sequence, head_dim].  The model
            # runner gives us flattened [tokens, heads, head_dim] rows; keep
            # the token axis intact instead of transposing it into head_dim.
            q = query[q_start:q_end].transpose(0, 1).unsqueeze(0)
            request_key = key[k_start:k_end]
            request_value = value[k_start:k_end]
            request_query_heads = query.shape[1]
            if request_key.shape[1] != request_query_heads:
                if request_query_heads % request_key.shape[1]:
                    raise ValueError(
                        "HyQuant dense prefill requires query heads to be a "
                        "multiple of KV heads"
                    )
                group_size = request_query_heads // request_key.shape[1]
                request_key = request_key.repeat_interleave(group_size, dim=1)
                request_value = request_value.repeat_interleave(group_size, dim=1)
            k = request_key.transpose(0, 1).unsqueeze(0)
            v = request_value.transpose(0, 1).unsqueeze(0)
            query_positions = context_len + torch.arange(query_len, device=query.device)
            key_positions = torch.arange(key_len, device=query.device)
            causal = key_positions[None, :] <= query_positions[:, None]
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal,
                dropout_p=0.0,
                scale=self.scale,
            )
            output[q_start:q_end].copy_(out.squeeze(0).transpose(0, 1))
        return output

    def _run_flash_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_start_loc: torch.Tensor,
        key_start_loc: torch.Tensor,
        query_start_loc_cpu: list[int],
        key_start_loc_cpu: list[int],
        max_query_len: int,
        max_key_len: int,
    ) -> torch.Tensor:
        if query.numel() == 0:
            return output
        if not is_flash_attn_varlen_func_available() or self.fa_version is None:
            return self._run_dense_prefill_fallback(
                query,
                key,
                value,
                output,
                query_start_loc_cpu,
                key_start_loc_cpu,
            )
        cu_q = query_start_loc.to(device=query.device, dtype=torch.int32)
        cu_k = key_start_loc.to(device=query.device, dtype=torch.int32)
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_key_len,
            softmax_scale=self.scale,
            causal=True,
            fa_version=self.fa_version,
        )
        return output

    def _run_mixed_prefix_prefill(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        suffix_keys: list[torch.Tensor],
        suffix_values: list[torch.Tensor],
        output: torch.Tensor,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: list[int],
        suffix_kv_start_loc: torch.Tensor,
        suffix_kv_start_loc_cpu: list[int],
        block_table: torch.Tensor,
        cached_prefix_lens: torch.Tensor,
        max_query_len: int,
        max_suffix_len: int,
    ) -> bool:
        """Run a cache-hit prefill without materializing compact pages.

        The direct Triton kernel computes the cached compact prefix partial and
        its LSE.  The BF16 suffix is still handled by FA2, then the two partial
        states are merged with the standard numerically stable LSE operation.
        """
        if len(suffix_keys) == 1:
            suffix_key = suffix_keys[0]
            suffix_value = suffix_values[0]
        else:
            suffix_key = torch.cat(suffix_keys, dim=0)
            suffix_value = torch.cat(suffix_values, dim=0)

        prefix_output = torch.empty_like(query)
        prefix_lse = torch.empty(
            query.shape[1], query.shape[0], dtype=torch.float32, device=query.device
        )
        prefix_result = triton_hyquant_prefix_attention(
            query,
            # The compact pages are read directly by the kernel.  No BF16
            # prefix workspace is allocated here.
            kv_cache,
            block_table,
            prefill_query_start_loc,
            cached_prefix_lens,
            self.block_layout,
            self.scale,
            max_query_len,
            output=prefix_output,
            lse=prefix_lse,
        )
        if prefix_result is None:
            return False

        suffix_output = torch.empty_like(query)
        suffix_lse = torch.empty(
            query.shape[1], query.shape[0], dtype=torch.float32, device=query.device
        )
        if not is_flash_attn_varlen_func_available() or self.fa_version is None:
            # This path is only for installations without FA2.  It keeps the
            # same semantics as the production split path for diagnostics.
            self._run_dense_prefill_fallback(
                query,
                suffix_key,
                suffix_value,
                suffix_output,
                prefill_query_start_loc_cpu,
                suffix_kv_start_loc_cpu,
            )
            self._compute_dense_lse(
                query,
                suffix_key,
                suffix_output,
                suffix_lse,
                prefill_query_start_loc_cpu,
                suffix_kv_start_loc_cpu,
            )
        else:
            result = flash_attn_varlen_func(
                q=query,
                k=suffix_key,
                v=suffix_value,
                out=suffix_output,
                cu_seqlens_q=prefill_query_start_loc,
                cu_seqlens_k=suffix_kv_start_loc,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_suffix_len,
                softmax_scale=self.scale,
                causal=True,
                return_softmax_lse=True,
                fa_version=self.fa_version,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("FA2 did not return suffix softmax LSE")
            suffix_output, suffix_lse = result

        merge_attn_states(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
        )
        return True

    def _compute_dense_lse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        query_start_loc_cpu: list[int],
        key_start_loc_cpu: list[int],
    ) -> None:
        """Compute suffix LSE for the no-FA2 diagnostic fallback."""
        del output
        query_heads = query.shape[1]
        kv_heads = key.shape[1]
        if query_heads % kv_heads:
            raise ValueError("HyQuant dense suffix requires divisible head counts")
        repeat = query_heads // kv_heads
        for request_index in range(len(query_start_loc_cpu) - 1):
            q_start = query_start_loc_cpu[request_index]
            q_end = query_start_loc_cpu[request_index + 1]
            k_start = key_start_loc_cpu[request_index]
            k_end = key_start_loc_cpu[request_index + 1]
            if q_end <= q_start:
                continue
            request_key = key[k_start:k_end].repeat_interleave(repeat, dim=1)
            q = query[q_start:q_end].transpose(0, 1).float()
            k = request_key.transpose(0, 1).float()
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            query_len = q_end - q_start
            key_len = k_end - k_start
            context_len = key_len - query_len
            positions_q = context_len + torch.arange(query_len, device=query.device)
            positions_k = torch.arange(key_len, device=query.device)
            scores = scores.masked_fill(
                positions_k[None, :] > positions_q[:, None], float("-inf")
            )
            lse[:, q_start:q_end] = torch.logsumexp(scores, dim=-1)

    def _validate_prefill_rows(
        self,
        query_start_loc_cpu: list[int],
        seq_lens_cpu: list[int],
        prompt_lens_cpu: list[int],
        cached_prefix_lens_cpu: list[int],
        state_slots_cpu: list[int],
    ) -> list[tuple[int, int, int, int, int, int, int]]:
        num_requests = len(seq_lens_cpu)
        if (
            len(query_start_loc_cpu) != num_requests + 1
            or len(prompt_lens_cpu) != num_requests
            or len(cached_prefix_lens_cpu) != num_requests
            or len(state_slots_cpu) != num_requests
        ):
            raise ValueError("HyQuant prefill metadata has inconsistent lengths")

        rows: list[tuple[int, int, int, int, int, int, int]] = []
        seen_slots: set[int] = set()
        for request_index in range(num_requests):
            q_start = int(query_start_loc_cpu[request_index])
            q_end = int(query_start_loc_cpu[request_index + 1])
            query_len = q_end - q_start
            sequence_length = int(seq_lens_cpu[request_index])
            prompt_length = int(prompt_lens_cpu[request_index])
            cached_prefix = int(cached_prefix_lens_cpu[request_index])
            state_slot = int(state_slots_cpu[request_index])
            context_length = sequence_length - query_len

            if cached_prefix < 0 or cached_prefix % self.block_size:
                raise RuntimeError(
                    "HyQuant cached prefixes must contain complete physical blocks"
                )
            if query_len <= 0 or context_length < 0:
                raise RuntimeError("HyQuant received invalid prefill chunk lengths")
            if cached_prefix > context_length:
                raise RuntimeError(
                    "HyQuant cached prefix exceeds the prefill context: "
                    f"{cached_prefix} > {context_length}"
                )
            if prompt_length <= 0 or sequence_length > prompt_length:
                raise RuntimeError(
                    "HyQuant prefill sequence length exceeds its fixed prompt length"
                )
            if state_slot < 0 or state_slot in seen_slots:
                raise RuntimeError("HyQuant received an invalid prefill state slot")
            seen_slots.add(state_slot)

            state = self._prefill_staging.get(state_slot)
            if context_length > cached_prefix:
                if state is None:
                    raise RuntimeError(
                        "HyQuant received a prefill continuation without its "
                        "earlier chunks"
                    )
                if state.prompt_len != prompt_length:
                    raise RuntimeError(
                        "HyQuant prompt length changed during chunked prefill"
                    )
                if state.cached_prefix_len != cached_prefix:
                    raise RuntimeError(
                        "HyQuant cached prefix changed during chunked prefill"
                    )
                if state.filled_len != context_length:
                    raise RuntimeError(
                        "HyQuant requires contiguous prefill chunks: expected "
                        f"offset {state.filled_len}, got {context_length}"
                    )
            rows.append(
                (
                    q_start,
                    q_end,
                    sequence_length,
                    prompt_length,
                    context_length,
                    cached_prefix,
                    state_slot,
                )
            )
        return rows

    def _prepare_chunked_prefill_kv(
        self,
        prefill_key: torch.Tensor,
        prefill_value: torch.Tensor,
        query_start_loc_cpu: list[int],
        seq_lens_cpu: list[int],
        prompt_lens_cpu: list[int],
        cached_prefix_lens_cpu: list[int],
        state_slots_cpu: list[int],
        rows: list[tuple[int, int, int, int, int, int, int]] | None = None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[int],
        list[int],
    ]:
        """Append current chunks and return uncached BF16 K/V suffixes."""
        if rows is None:
            rows = self._validate_prefill_rows(
                query_start_loc_cpu,
                seq_lens_cpu,
                prompt_lens_cpu,
                cached_prefix_lens_cpu,
                state_slots_cpu,
            )
        attention_keys: list[torch.Tensor] = []
        attention_values: list[torch.Tensor] = []
        completed_requests: list[int] = []
        completed_staged_slots: list[int] = []

        for request_index, row in enumerate(rows):
            (
                q_start,
                q_end,
                sequence_length,
                prompt_length,
                context_length,
                cached_prefix,
                slot,
            ) = row
            current_key = prefill_key[q_start:q_end]
            current_value = prefill_value[q_start:q_end]
            if context_length == cached_prefix:
                # A stable slot may have belonged to an aborted/preempted
                # request. The initial uncached position defines a new epoch.
                self._prefill_staging.pop(slot, None)
                if sequence_length == prompt_length:
                    full_key = current_key
                    full_value = current_value
                else:
                    suffix_capacity = prompt_length - cached_prefix
                    state = _HyQuantPrefillState(
                        key=torch.empty(
                            (suffix_capacity, *current_key.shape[1:]),
                            dtype=current_key.dtype,
                            device=current_key.device,
                        ),
                        value=torch.empty(
                            (suffix_capacity, *current_value.shape[1:]),
                            dtype=current_value.dtype,
                            device=current_value.device,
                        ),
                        filled_len=sequence_length,
                        prompt_len=prompt_length,
                        cached_prefix_len=cached_prefix,
                    )
                    suffix_length = sequence_length - cached_prefix
                    state.key[:suffix_length].copy_(current_key)
                    state.value[:suffix_length].copy_(current_value)
                    self._prefill_staging[slot] = state
                    full_key = state.key[:suffix_length]
                    full_value = state.value[:suffix_length]
            else:
                state = self._prefill_staging[slot]
                expected_shape = (
                    prompt_length - cached_prefix,
                    prefill_key.shape[1],
                    prefill_key.shape[2],
                )
                if (
                    state.key.shape != expected_shape
                    or state.value.shape
                    != (
                        prompt_length - cached_prefix,
                        prefill_value.shape[1],
                        prefill_value.shape[2],
                    )
                    or state.key.dtype != prefill_key.dtype
                    or state.value.dtype != prefill_value.dtype
                    or state.key.device != prefill_key.device
                    or state.value.device != prefill_value.device
                ):
                    raise RuntimeError(
                        "HyQuant prefill continuation K/V does not match its "
                        "staged prompt"
                    )
                relative_context = context_length - cached_prefix
                relative_sequence = sequence_length - cached_prefix
                state.key[relative_context:relative_sequence].copy_(current_key)
                state.value[relative_context:relative_sequence].copy_(current_value)
                state.filled_len = sequence_length
                full_key = state.key[:relative_sequence]
                full_value = state.value[:relative_sequence]

            attention_keys.append(full_key)
            attention_values.append(full_value)
            if sequence_length == prompt_length:
                completed_requests.append(request_index)
                if slot in self._prefill_staging:
                    completed_staged_slots.append(slot)

        return (
            attention_keys,
            attention_values,
            completed_requests,
            completed_staged_slots,
        )

    def _release_completed_prefills(self, state_slots: list[int]) -> None:
        for state_slot in state_slots:
            self._prefill_staging.pop(state_slot, None)

    def _pack_new_prefill_blocks(
        self,
        suffix_keys: list[torch.Tensor],
        suffix_values: list[torch.Tensor],
        prefill_query: torch.Tensor,
        query_start_loc_cpu: list[int],
        rows: list[tuple[int, int, int, int, int, int, int]],
        block_table: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> None:
        """Pack cacheable blocks completed by this prefill chunk.

        Every block uses its own final query for anchor selection. This makes a
        page independent of later prompt tokens and safe to share through APC.
        """
        source_keys: list[torch.Tensor] = []
        source_values: list[torch.Tensor] = []
        physical_ids: list[torch.Tensor] = []
        anchor_indices: list[torch.Tensor] = []
        requests = zip(suffix_keys, suffix_values, rows, strict=True)
        for request_index, (suffix_key, suffix_value, row) in enumerate(requests):
            (
                q_start,
                q_end,
                sequence_length,
                prompt_length,
                context_length,
                cached_prefix,
                _,
            ) = row
            cacheable_prefix = (
                max(prompt_length - self.window_size, 0)
                // self.block_size
                * self.block_size
            )
            if cached_prefix > cacheable_prefix:
                raise RuntimeError(
                    "HyQuant prefix-cache hit extends into the private BF16 tail"
                )
            first_block, end_block = get_hyquant_new_completed_block_range(
                context_length, sequence_length, self.block_size
            )
            first_block = max(first_block, cached_prefix // self.block_size)
            end_block = min(end_block, cacheable_prefix // self.block_size)
            if end_block <= first_block:
                continue

            first_position = first_block * self.block_size
            end_position = end_block * self.block_size
            suffix_start = first_position - cached_prefix
            suffix_end = end_position - cached_prefix
            if suffix_start < 0 or suffix_end > suffix_key.shape[0]:
                raise RuntimeError("HyQuant prefill staging does not cover a new block")
            source_key = suffix_key[suffix_start:suffix_end].reshape(
                end_block - first_block,
                self.block_size,
                self.num_kv_heads,
                self.head_size,
            )
            source_value = suffix_value[suffix_start:suffix_end].reshape_as(source_key)
            first_query_offset = first_position + self.block_size - 1 - context_length
            last_query_offset = end_position - 1 - context_length
            if first_query_offset < 0 or last_query_offset >= q_end - q_start:
                raise RuntimeError("HyQuant block-final prefill query is unavailable")
            query_offsets = torch.arange(
                first_query_offset,
                last_query_offset + 1,
                self.block_size,
                dtype=torch.long,
                device=prefill_query.device,
            )
            reference_query = prefill_query[q_start + query_offsets]
            indices = select_important_token_indices(
                source_key.reshape(-1, self.num_kv_heads, self.head_size),
                reference_query,
                self.block_size,
                self.top_ratio,
            )
            source_keys.append(source_key)
            source_values.append(source_value)
            physical_ids.append(block_table[request_index, first_block:end_block])
            anchor_indices.append(indices)
        if not source_keys:
            return
        if len(source_keys) == 1:
            packed_keys = source_keys[0]
            packed_values = source_values[0]
            packed_ids = physical_ids[0]
            packed_anchors = anchor_indices[0]
        else:
            packed_keys = torch.cat(source_keys, dim=0)
            packed_values = torch.cat(source_values, dim=0)
            packed_ids = torch.cat(physical_ids, dim=0)
            packed_anchors = torch.cat(anchor_indices, dim=0)
        pack_hyquant_blocks(
            packed_keys,
            packed_values,
            packed_ids,
            packed_anchors,
            kv_cache,
            self.block_layout,
        )

    def _retire_decode_blocks(
        self,
        decode_query: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        prompt_lens_cpu: torch.Tensor,
        state_slots: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> None:
        """Move blocks leaving the BF16 window into compact mixed pages."""
        source_keys: list[torch.Tensor] = []
        source_values: list[torch.Tensor] = []
        physical_ids: list[torch.Tensor] = []
        anchor_indices: list[torch.Tensor] = []
        for request_index, sequence_length in enumerate(seq_lens_cpu):
            sequence_length = int(sequence_length)
            prompt_length = int(prompt_lens_cpu[request_index])
            previous_length = max(sequence_length - 1, prompt_length)
            previous_committed = get_hyquant_committed_len(
                previous_length,
                prompt_length,
                self.block_size,
                self.window_size,
                self.retire_interval,
            )
            committed = get_hyquant_committed_len(
                sequence_length,
                prompt_length,
                self.block_size,
                self.window_size,
                self.retire_interval,
            )
            if committed <= previous_committed:
                continue
            slot = int(state_slots[request_index])
            first_block = previous_committed // self.block_size
            last_block = committed // self.block_size
            positions = []
            for logical_block in range(first_block, last_block):
                logical_positions = torch.arange(
                    logical_block * self.block_size,
                    (logical_block + 1) * self.block_size,
                    device=kv_cache.device,
                )
                ring = logical_positions.remainder(self.hot_capacity)
                positions.append(ring)
            if not positions:
                continue
            ring_positions = torch.cat(positions)
            source_k = (
                self.hot_key[slot].index_select(1, ring_positions).permute(1, 0, 2)
            )
            source_v = (
                self.hot_value[slot].index_select(1, ring_positions).permute(1, 0, 2)
            )
            num_blocks = last_block - first_block
            indices = select_important_token_indices(
                source_k,
                decode_query[request_index],
                self.block_size,
                self.top_ratio,
            )
            source_keys.append(
                source_k.view(
                    num_blocks, self.block_size, self.num_kv_heads, self.head_size
                )
            )
            source_values.append(
                source_v.view(
                    num_blocks, self.block_size, self.num_kv_heads, self.head_size
                )
            )
            physical_ids.append(block_table[request_index, first_block:last_block])
            anchor_indices.append(indices)
        if source_keys:
            pack_hyquant_blocks(
                torch.cat(source_keys, dim=0),
                torch.cat(source_values, dim=0),
                torch.cat(physical_ids, dim=0),
                torch.cat(anchor_indices, dim=0),
                kv_cache,
                self.block_layout,
            )

    def _retire_decode_blocks_gpu(
        self,
        decode_query: torch.Tensor,
        decode_key: torch.Tensor,
        decode_value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        state_slots: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> bool:
        """Launch graph-safe selection/packing for pure one-token decode."""
        if not self._gpu_retire_enabled:
            return False
        return triton_hyquant_retire_blocks(
            decode_query,
            self.hot_key,
            self.hot_value,
            kv_cache,
            block_table,
            seq_lens,
            prompt_lens,
            state_slots,
            self._retire_anchor_workspace,
            self.block_layout,
            self.window_size,
            self.retire_interval,
            self._retire_block_table_width,
        )

    def _run_int8_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        hot_key: torch.Tensor,
        hot_value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        state_slots: torch.Tensor,
        output: torch.Tensor,
        mid_o_buf: torch.Tensor | None,
        max_seq_len: int,
        current_key: torch.Tensor | None,
        current_value: torch.Tensor | None,
        fuse_current_store: bool,
        *,
        decode_num_splits: int | None = None,
        decode_block_kv: int | None = None,
        decode_block_h: int | None = None,
        decode_num_warps: int | None = None,
        decode_num_stages: int | None = None,
        decode_use_simt: bool | None = None,
    ) -> torch.Tensor | None:
        """Launch the low-overhead fixed-shape decoder when it is applicable.

        This deliberately contains only dispatch checks.  The generic
        ``triton_hyquant_block_decode_attention`` entry point performs much
        broader ABI validation for fallback layouts; repeating that work for
        every layer is measurable in eager decode.
        """
        if not self._int8_decode_enabled:
            return None
        if (
            query.ndim != 3
            or query.shape[1] != self.num_heads
            or query.shape[-1] != 128
            or query.dtype != torch.bfloat16
            or kv_cache.dtype != torch.uint8
            or kv_cache.ndim != 3
            or kv_cache.shape[1] != self.num_kv_heads
            or query.shape[0] <= 0
        ):
            return None
        batch_size = int(query.shape[0])
        if max_seq_len <= 0:
            max_seq_len = max(1, int(seq_lens.max().item()))
        num_splits = (
            int(decode_num_splits)
            if decode_num_splits is not None
            else (get_hyquant_decode_split_count(batch_size, max_seq_len, 4))
        )
        if num_splits > 1 and (
            mid_o_buf is None
            or query.shape[0] > mid_o_buf.shape[0]
            or mid_o_buf.dtype != torch.float32
            or mid_o_buf.device != query.device
        ):
            return None
        if (
            decode_block_kv is None
            or decode_block_h is None
            or decode_num_warps is None
        ):
            block_kv, block_h, num_warps = get_hyquant_decode_launch_config(
                batch_size, max_seq_len, 4
            )
        else:
            block_kv, block_h, num_warps = (
                int(decode_block_kv),
                int(decode_block_h),
                int(decode_num_warps),
            )
        if (
            num_splits <= 0
            or block_h != 4
            or block_kv > 128
            or (
                num_splits > 1
                and (
                    mid_o_buf is None
                    or mid_o_buf.ndim != 4
                    or mid_o_buf.shape[1] < self.num_heads
                    or mid_o_buf.shape[2] < num_splits
                    or mid_o_buf.shape[3] < 129
                )
            )
        ):
            return None
        if fuse_current_store and (
            current_key is None
            or current_value is None
            or current_key.shape != (batch_size, self.num_kv_heads, 128)
            or current_value.shape != (batch_size, self.num_kv_heads, 128)
        ):
            return None
        try:
            return hyquant_int8qk_decode(
                query,
                kv_cache,
                hot_key,
                hot_value,
                block_table,
                seq_lens,
                prompt_lens,
                state_slots,
                self.block_layout,
                self.window_size,
                self.retire_interval,
                self.scale,
                (
                    None
                    if num_splits == 1
                    else mid_o_buf[:batch_size, : self.num_heads, :num_splits, :129]
                ),
                num_splits,
                block_kv,
                block_h,
                num_warps,
                (
                    get_hyquant_decode_num_stages(batch_size, max_seq_len)
                    if decode_num_stages is None
                    else int(decode_num_stages)
                ),
                use_simt=(False if decode_use_simt is None else bool(decode_use_simt)),
                output=output,
                current_key=current_key,
                current_value=current_value,
                fuse_current_store=fuse_current_store,
            )
        except Exception:
            if getattr(self, "_grouped_strict", False):
                raise
            return None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HyQuantMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run BF16 FA2 prefill followed by block-typed cache attention."""
        del layer, output_scale, output_block_scale
        if output is None:
            output = torch.empty_like(query)
        if attn_metadata is None:
            return output.zero_()
        num_tokens = min(attn_metadata.num_actual_tokens, query.shape[0])
        if num_tokens <= 0:
            return output.zero_()
        if key is None or value is None:
            raise ValueError("HyQuant requires current K/V for cache update")
        if (
            query.dtype != torch.bfloat16
            or key.dtype != torch.bfloat16
            or value.dtype != torch.bfloat16
        ):
            raise ValueError(
                "HyQuant block layout v1 stores full-precision pages as BF16; "
                "use --dtype bfloat16."
            )
        query = query[:num_tokens]
        key = key[:num_tokens]
        value = value[:num_tokens]
        if kv_cache.shape[1] != self.num_kv_heads:
            raise ValueError("HyQuant cache KV-head count does not match the layer")
        num_decode_tokens = attn_metadata.num_decode_tokens
        has_prefill = num_decode_tokens < num_tokens

        if num_decode_tokens and num_decode_tokens != attn_metadata.num_decodes:
            raise RuntimeError(
                "HyQuant block layout v1 currently supports one decode token "
                "per request"
            )

        if has_prefill:
            prefill_query = query[num_decode_tokens:]
            prefill_key = key[num_decode_tokens:]
            prefill_value = value[num_decode_tokens:]
            global_qsl = attn_metadata.query_start_loc_cpu.tolist()
            local_qsl = [
                int(x - num_decode_tokens)
                for x in global_qsl[attn_metadata.num_decodes :]
            ]
            num_prefill_requests = len(local_qsl) - 1
            request_slice = slice(
                attn_metadata.num_decodes,
                attn_metadata.num_decodes + num_prefill_requests,
            )
            prefill_block_table = attn_metadata.block_table[request_slice]
            seq_cpu = [
                int(x) for x in attn_metadata.seq_lens_cpu[request_slice].tolist()
            ]
            cached_cpu = [
                int(x)
                for x in attn_metadata.cached_prefix_lens_cpu[request_slice].tolist()
            ]
            prompt_cpu = [
                int(x) for x in attn_metadata.prompt_lens_cpu[request_slice].tolist()
            ]
            state_slots_cpu = [
                int(x) for x in attn_metadata.state_slots_cpu[request_slice].tolist()
            ]
            rows = self._validate_prefill_rows(
                local_qsl,
                seq_cpu,
                prompt_cpu,
                cached_cpu,
                state_slots_cpu,
            )
            (
                suffix_keys,
                suffix_values,
                _,
                completed_staged_slots,
            ) = self._prepare_chunked_prefill_kv(
                prefill_key,
                prefill_value,
                local_qsl,
                seq_cpu,
                prompt_cpu,
                cached_cpu,
                state_slots_cpu,
                rows=rows,
            )

            # vLLM can publish a writer's full blocks to the local prefix cache
            # before this model step executes. Pack every writer first so a
            # same-step cache-hit reader always sees initialized compact pages.
            self._pack_new_prefill_blocks(
                suffix_keys,
                suffix_values,
                prefill_query,
                local_qsl,
                rows,
                prefill_block_table,
                kv_cache,
            )

            direct_mixed_prefill = False
            if any(cached_cpu):
                direct_mixed_prefill = self._run_mixed_prefix_prefill(
                    prefill_query,
                    kv_cache,
                    suffix_keys,
                    suffix_values,
                    output[num_decode_tokens:num_tokens],
                    attn_metadata.prefill_query_start_loc,
                    local_qsl,
                    attn_metadata.prefill_suffix_kv_start_loc,
                    [
                        int(x)
                        for x in attn_metadata.prefill_suffix_kv_start_loc_cpu.tolist()
                    ],
                    prefill_block_table,
                    attn_metadata.cached_prefix_lens[request_slice],
                    attn_metadata.max_prefill_query_len,
                    max(
                        (sequence_length - cached_prefix)
                        for sequence_length, cached_prefix in zip(
                            seq_cpu, cached_cpu, strict=True
                        )
                    ),
                )

            if direct_mixed_prefill:
                pass
            elif not any(cached_cpu):
                if len(suffix_keys) == 1:
                    attention_key = suffix_keys[0]
                    attention_value = suffix_values[0]
                else:
                    attention_key = torch.cat(suffix_keys, dim=0)
                    attention_value = torch.cat(suffix_values, dim=0)
            else:
                raise RuntimeError(
                    "HyQuant direct compact-prefix prefill is unavailable for "
                    "this device/layout; refusing the phase-one BF16 materialize "
                    "fallback. Check Triton/CUDA support or disable HyQuant."
                )
            if not direct_mixed_prefill:
                kv_qsl_cpu = [
                    int(x) for x in attn_metadata.prefill_kv_start_loc_cpu.tolist()
                ]
                if kv_qsl_cpu[-1] != attention_key.shape[0]:
                    raise RuntimeError(
                        "HyQuant accumulated prefill K/V length does not match "
                        "scheduler metadata"
                    )
                self._run_flash_prefill(
                    prefill_query,
                    attention_key,
                    attention_value,
                    output[num_decode_tokens:num_tokens],
                    attn_metadata.prefill_query_start_loc,
                    attn_metadata.prefill_kv_start_loc,
                    local_qsl,
                    kv_qsl_cpu,
                    attn_metadata.max_prefill_query_len,
                    attn_metadata.max_prefill_seq_len,
                )
            self._release_completed_prefills(completed_staged_slots)
            if not triton_hyquant_store_hot(
                prefill_key,
                prefill_value,
                self.hot_key,
                self.hot_value,
                attn_metadata.slot_mapping[num_decode_tokens:num_tokens],
                attn_metadata.positions[num_decode_tokens:num_tokens],
                attn_metadata.token_to_req[num_decode_tokens:num_tokens],
                attn_metadata.state_slots,
                attn_metadata.seq_lens,
                attn_metadata.prompt_lens,
                self.block_size,
                self.window_size,
                self.retire_interval,
                cached_prefix_lens=attn_metadata.cached_prefix_lens,
                is_prefill=True,
            ):
                raise RuntimeError("HyQuant BF16 window store was unavailable")

        if num_decode_tokens:
            decode_query = query[:num_decode_tokens]
            decode_key = key[:num_decode_tokens]
            decode_value = value[:num_decode_tokens]
            # CPU lengths are already present in the shared metadata.  Keep
            # them as tensor views; converting to Python lists here used to
            # happen once per layer and dominated short decode steps.
            seq_cpu = attn_metadata.seq_lens_cpu[: attn_metadata.num_decodes]
            prompt_cpu = attn_metadata.prompt_lens_cpu[: attn_metadata.num_decodes]
            decode_block_table = attn_metadata.block_table[: attn_metadata.num_decodes]
            decode_seq_lens = attn_metadata.seq_lens[: attn_metadata.num_decodes]
            decode_prompt_lens = attn_metadata.prompt_lens[: attn_metadata.num_decodes]
            decode_state_slots = attn_metadata.state_slots[: attn_metadata.num_decodes]
            decode_slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens]
            decode_positions = attn_metadata.positions[:num_decode_tokens]
            decode_token_to_req = attn_metadata.token_to_req[:num_decode_tokens]

            # A pure one-token decode has the current K/V for every request in
            # a compact [B, H_kv, D] tensor.  Fuse its hot-ring append into the
            # grouped attention launch.  Retirement must happen first because
            # it reads the previous hot-ring contents; the fused kernel then
            # publishes the new row after consuming it for attention.
            fuse_current_store = (
                self._fuse_current_store_enabled
                and num_decode_tokens == attn_metadata.num_decodes
                and decode_key.shape[0] == attn_metadata.num_decodes
            )
            # V2 Model Runner computes the boundary on the host before dispatch
            # and forces that one step to eager execution.  Consequently the
            # metadata flag is reliable here: graph-captured ordinary decode
            # has ``retirement_needed=False`` and should not pay a per-layer
            # no-op retirement launch, while the eager boundary step sets it
            # and packs pages before attention reads them.  Keep the former
            # graph-safe every-step behavior available only as an explicit A/B
            # switch because it is useful for older runners that do not provide
            # the boundary dispatch contract.
            every_step_retirement = self._every_step_retirement_enabled
            gpu_retirement = False
            if (
                num_decode_tokens == attn_metadata.num_decodes
                and decode_key.shape[0] == attn_metadata.num_decodes
                and self._gpu_retire_enabled
                and (
                    attn_metadata.retirement_needed
                    or every_step_retirement
                    # During CUDA-graph capture the metadata contains a
                    # conservative non-boundary length.  Still record the
                    # graph-safe retirement launch; its device-side active
                    # predicate makes ordinary replay steps a cheap no-op.
                    or (
                        self._graph_retire_enabled
                        and torch.cuda.is_current_stream_capturing()
                    )
                )
            ):
                try:
                    gpu_retirement = self._retire_decode_blocks_gpu(
                        decode_query,
                        decode_key,
                        decode_value,
                        decode_block_table,
                        decode_seq_lens,
                        decode_prompt_lens,
                        decode_state_slots,
                        kv_cache,
                    )
                except Exception:
                    if self._grouped_strict:
                        raise
                    logger.warning(
                        "HyQuant GPU retirement failed; using host fallback for "
                        "this worker. Set VLLM_HYQUANT_GROUPED_STRICT=1 to raise.",
                        exc_info=True,
                    )
                    self._gpu_retire_enabled = False
            if (
                fuse_current_store
                and not gpu_retirement
                and attn_metadata.retirement_needed
            ):
                self._retire_decode_blocks(
                    decode_query,
                    decode_block_table,
                    seq_cpu,
                    prompt_cpu,
                    decode_state_slots,
                    kv_cache,
                )

            def store_decode_hot() -> None:
                if not triton_hyquant_store_hot(
                    decode_key,
                    decode_value,
                    self.hot_key,
                    self.hot_value,
                    decode_slot_mapping,
                    decode_positions,
                    decode_token_to_req,
                    attn_metadata.state_slots,
                    attn_metadata.seq_lens,
                    attn_metadata.prompt_lens,
                    self.block_size,
                    self.window_size,
                    self.retire_interval,
                    cached_prefix_lens=attn_metadata.cached_prefix_lens,
                    is_prefill=False,
                ):
                    raise RuntimeError("HyQuant BF16 window store was unavailable")

            if not fuse_current_store:
                store_decode_hot()
                if not gpu_retirement and attn_metadata.retirement_needed:
                    self._retire_decode_blocks(
                        decode_query,
                        decode_block_table,
                        seq_cpu,
                        prompt_cpu,
                        decode_state_slots,
                        kv_cache,
                    )

            decode_workspace = attn_metadata.mid_o_buf
            if decode_workspace is None and attn_metadata.decode_num_splits > 1:
                decode_workspace = self._get_decode_workspace(num_decode_tokens)
            decode_output = self._run_int8_decode(
                decode_query,
                kv_cache,
                self.hot_key,
                self.hot_value,
                decode_block_table,
                decode_seq_lens,
                decode_prompt_lens,
                decode_state_slots,
                output[:num_decode_tokens],
                decode_workspace,
                attn_metadata.max_decode_seq_len,
                decode_key,
                decode_value,
                fuse_current_store,
                decode_num_splits=attn_metadata.decode_num_splits,
                decode_block_kv=attn_metadata.decode_block_kv,
                decode_block_h=attn_metadata.decode_block_h,
                decode_num_warps=attn_metadata.decode_num_warps,
                decode_num_stages=attn_metadata.decode_num_stages,
                decode_use_simt=attn_metadata.decode_use_simt,
            )
            if decode_output is None:
                decode_output = triton_hyquant_block_decode_attention(
                    decode_query,
                    kv_cache,
                    self.hot_key,
                    self.hot_value,
                    decode_block_table,
                    decode_seq_lens,
                    decode_prompt_lens,
                    decode_state_slots,
                    self.block_layout,
                    self.window_size,
                    self.retire_interval,
                    self.scale,
                    output=output[:num_decode_tokens],
                    mid_o_buf=decode_workspace,
                    max_seq_len=attn_metadata.max_decode_seq_len,
                    current_key=decode_key,
                    current_value=decode_value,
                    fuse_current_store=fuse_current_store,
                )
            if decode_output is None and fuse_current_store:
                # Compilation or a target-specific ABI rejection must not
                # leave the current row unpublished.  Retry with the explicit
                # store and the validated non-fused grouped path.
                store_decode_hot()
                decode_output = triton_hyquant_block_decode_attention(
                    decode_query,
                    kv_cache,
                    self.hot_key,
                    self.hot_value,
                    decode_block_table,
                    decode_seq_lens,
                    decode_prompt_lens,
                    decode_state_slots,
                    self.block_layout,
                    self.window_size,
                    self.retire_interval,
                    self.scale,
                    output=output[:num_decode_tokens],
                    mid_o_buf=decode_workspace,
                    max_seq_len=attn_metadata.max_decode_seq_len,
                )
            if decode_output is None:
                hyquant_block_decode_attention(
                    decode_query,
                    kv_cache,
                    self.hot_key,
                    self.hot_value,
                    decode_block_table,
                    decode_seq_lens,
                    decode_prompt_lens,
                    decode_state_slots,
                    self.block_layout,
                    self.window_size,
                    self.retire_interval,
                    self.scale,
                    output=output[:num_decode_tokens],
                )
        return output

    def _get_decode_workspace(self, batch_size: int) -> torch.Tensor | None:
        """Return a reusable grouped split-K workspace for one decode batch."""
        if not is_workspace_manager_initialized() or batch_size <= 0:
            return None
        manager = current_workspace_manager()
        block_d = self._decode_workspace_shape[3]
        # Keep the shape at the configured maximum split count.  The kernel
        # slices the first N splits, while the workspace manager reuses the
        # same allocation across all layers and ubatches.
        try:
            (workspace,) = manager.get_simultaneous(
                (
                    (
                        batch_size,
                        self.num_heads,
                        self._max_decode_splits,
                        block_d + 1,
                    ),
                    torch.float32,
                )
            )
        except AssertionError:
            # A locked manager can legitimately be smaller when the deployment
            # never captured a split-K shape.  Returning None lets the launcher
            # use its eager fallback rather than mutating a locked workspace.
            return None
        return workspace
