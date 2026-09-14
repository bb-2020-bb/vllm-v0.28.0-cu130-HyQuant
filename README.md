<!-- markdownlint-disable MD001 MD041 -->

## HyQuant Research Extension

This checkout contains an experimental **HyQuant KV-cache backend for vLLM
V1**. It keeps block-local important tokens and a recent token window in BF16,
stores the remaining K/V values as symmetric groupwise INT4, and consumes the
mixed compact pages directly during decode. The model weights and prefill
attention are not quantized by this feature: cold prefill continues to use
BF16 FlashAttention 2, followed by compact-page construction.

### Reproducible Environment

Use a fresh Linux environment with an NVIDIA GPU of compute capability 8.0 or
newer. The reference stack is Python 3.11, PyTorch 2.13.0+cu130, CUDA 13.0,
Triton 3.7.1, and vLLM 0.28.0. CUDA 13 normally requires an R580 or newer NVIDIA
driver; supported data-center GPUs can also use NVIDIA's CUDA forward
compatibility package with an eligible older driver.

The HyQuant changes in this repository are Python and Triton code, so the
recommended setup reuses vLLM's matching precompiled CUDA binaries and installs
this checkout in editable mode:

```bash
# Install uv first if it is not already available.
curl -LsSf https://astral.sh/uv/install.sh | sh

cd vllm-v0.28.0-cu130
uv venv --python 3.11 --seed
source .venv/bin/activate

export VLLM_PRECOMPILED_WHEEL_VARIANT=cu130
VLLM_USE_PRECOMPILED=1 \
  uv pip install --editable . --torch-backend=cu130
```

If a matching precompiled base wheel is unavailable, install the CUDA 13.0
toolkit and GCC/G++ 11.3 or newer, then replace the final command with
`uv pip install --editable . --torch-backend=cu130` for a full source build.
On CUDA, the reference configuration uses the FA2 extension bundled with vLLM;
a separate PyPI `flash-attn` installation is not required.

Verify the installation before serving a model:

```bash
python - <<'PY'
import torch
import triton
import vllm
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() >= (8, 0)
print("vLLM:", vllm.__version__)
print("PyTorch/CUDA:", torch.__version__, torch.version.cuda)
print("Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name())
print("FlashAttention version:", get_flash_attn_version(head_size=128))
PY
```

Current support includes decoder-only MHA/GQA/MQA, one-shot and chunked
prefill, mixed and multi-request decode, periodic KV retirement, local
in-process full-block prefix caching, both V2 and legacy V1 GPU model runners,
and normal single-token decode CUDA graphs. Speculative decoding, KV
offload/connectors,
cross-process prefix caches, partial-block prefix hits, DCP/PCP, MLA,
model-level sliding-window attention, ALiBi, and cross-attention are not
supported.

```bash
vllm serve /path/to/a/decoder-only-model \
  --dtype bfloat16 \
  --attention-backend HYQUANT \
  --attention-config.flash_attn_version=2 \
  --kv-cache-dtype hyquant_k4v4 \
  --block-size 16 \
  --hyquant-top-ratio 0.0625 \
  --hyquant-group-size 32 \
  --hyquant-window-size 256 \
  --hyquant-retire-interval 64
```

This is a research implementation whose primary measured benefit is roughly
`2.86x-3.0x` more persistent KV token capacity. It is not universally faster
than the BF16 FlashAttention 2 baseline. See the
[detailed Chinese implementation guide](HYQUANT_VLLM_IMPLEMENTATION_ZH.md) for
the layout, execution paths, source changes, validation, results, and limits.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
