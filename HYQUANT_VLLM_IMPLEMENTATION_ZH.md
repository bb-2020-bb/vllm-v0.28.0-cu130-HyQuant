# HyQuant 在 vLLM V1 中的实现与适配说明

> 文档范围：本文描述 `/home/zy/xbx/vllm-v0.28.0-cu130` 当前工作树中的
> HyQuant 活跃实现，基于 vLLM 0.28.0、基线提交
> `2cf0a6915ce544dc493a0990f2ea38d81601128a`，截至 2026-09-14 的 v51
> 状态。历史实验中的旧布局、失败方案和旧性能不能当作当前实现。

## 1. 一句话结论

这个分支把 HyQuant 的“重要 token 与局部窗口保留高精度，其余 KV 使用低比特”
思想改造成了一个原生 vLLM V1 attention backend：

- 模型权重、Q/K/V 激活和普通 prefill attention 仍为 BF16；
- 已经离开热窗口的 KV 按 vLLM 物理 block 存入固定大小的紧凑页；
- 每个 block 内固定数量的重要 token 保留 BF16 K/V；
- 其余 token 的 K 和 V 分别使用 symmetric groupwise INT4；
- decode kernel 直接消费 BF16 anchor、INT4 冷数据和 BF16 热窗口，不构造完整的
  BF16 历史 KV；
- 支持 chunked prefill、多请求 decode 和本地完整 block Prefix Cache；
- 默认布局的理论页压缩比约为 `3.01x`，真实模型启动时测得约
  `2.86x-3.0x` 的 KV token capacity；
- 当前主要收益是 KV 容量，不是所有场景都比 FA2 更快。长上下文 attention 核心
  已接近 FA2，甚至在一个 31.8K 单层测量中略快，但 backend 包装和端到端仍可能
  慢于 FA2，短上下文差距更明显。

这里的 HyQuant 只量化 **KV cache**，不是权重量化，也不是把整个模型推理改成
INT4。

## 2. 当前版本与验证环境

当前正式对比基线始终是 vLLM `FLASH_ATTN`，在这套环境中解析为
FlashAttention 2，而不是 `TRITON_ATTN`。

| 项目 | 当前验证值 |
| --- | --- |
| 仓库 | `/home/zy/xbx/vllm-v0.28.0-cu130` |
| vLLM | 0.28.0 |
| Conda 环境 | `sinkatt` |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| CUDA | 13.0 |
| Triton | 3.7.1 |
| flash-attn | 2.8.3 |
| 主要 GPU | 物理 GPU1，NVIDIA A800 80GB PCIe，SM80 |
| 主要模型 | `/home/zy/model/Qwen3-4B`，BF16，TP=1 |
| CUDA 兼容库 | `/usr/local/cuda-13.0/compat:/usr/local/cuda/lib64` |

典型环境设置：

```bash
conda activate sinkatt
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda/lib64
export PYTHONPATH=/home/zy/xbx/vllm-v0.28.0-cu130
cd /home/zy/xbx/vllm-v0.28.0-cu130
```

典型服务启动方式：

```bash
vllm serve /home/zy/model/Qwen3-4B \
  --dtype bfloat16 \
  --attention-backend HYQUANT \
  --kv-cache-dtype hyquant_k4v4 \
  --block-size 16 \
  --hyquant-top-ratio 0.0625 \
  --hyquant-group-size 32 \
  --hyquant-window-size 256 \
  --hyquant-retire-interval 64 \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

等价的 Python 核心配置为：

```python
from vllm import LLM

llm = LLM(
    model="/home/zy/model/Qwen3-4B",
    dtype="bfloat16",
    attention_backend="HYQUANT",
    kv_cache_dtype="hyquant_k4v4",
    block_size=16,
    hyquant_top_ratio=0.0625,
    hyquant_group_size=32,
    hyquant_window_size=256,
    hyquant_retire_interval=64,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)
```

以上是验证配置，不表示所有合法参数组合都完成了同等强度的模型级验证。

## 3. 原版 HyQuant 思想与本实现的关系

原版 `/home/zy/xbx/HyQuant` 观察到 attention map 中存在持续被关注的“vertical
line” token。它维护跨 decode step 的列注意力质量分数，保留全局 top-ratio token
和局部窗口，并为 Hugging Face `past_key_values` 提供混合精度 KV cache。

直接把原版 `DStageKVCache` 搬进 vLLM 不合适，原因是 vLLM 的 KV 生命周期由
PagedAttention、block table、scheduler、preemption、copy-on-write 和 Prefix
Cache 共同管理。任意跨 block 搬动 token 或使用可变大小页面，会破坏物理 block
和逻辑位置之间的稳定关系。

因此当前实现保留了 HyQuant 的核心目标，但为 vLLM 做了以下取舍：

| 方面 | 原版 HyQuant | 当前 vLLM 适配 |
| --- | --- | --- |
| KV 容器 | Hugging Face 自定义 cache | vLLM V1 固定大小 PagedAttention page |
| 重要 token | 跨序列运行中的 vertical-line top-k | 每个物理 block 内固定 top-A |
| 重要性更新 | 周期性累计/更新全局列质量 | block 完成或退休时选择一次，之后静态 |
| 最近窗口 | FP16/BF16 | 独立的 request-local BF16 circular ring |
| Prefill 计算 | 可使用混合低精度 kernel | 冷 prefill 保持 BF16 FA2，之后再 pack |
| Decode | 混合 KV fused attention | Triton 直接读 compact page + BF16 hot ring |
| Prefix Cache | 不是主要目标 | 完整物理 block、本地同进程、共享页只读 |

这不是论文代码逐行移植，而是以 vLLM 的 page、调度和共享语义为约束的工程化
变体。最重要的区别是 **block-local 固定 anchor 数量**。这个决定牺牲了全局
top-k 的自由度，但换来了固定 page stride、可靠的 allocator accounting，以及
无需改变 `block_table` 的 O(1) 定位。

## 4. 总体架构

### 4.1 三类 KV 状态

对任意一个正在生成的请求，KV 被分成三类：

1. **冷区 INT4 token**：已经离开热窗口，K/V 都是 groupwise INT4；
2. **冷区 BF16 anchor**：与 INT4 token 位于同一个紧凑物理 page 中；
3. **热区 BF16 token**：最近 `window_size` 左右的 token，位于独立 BF16 ring。

逻辑 token 顺序从未改变。`block_table[request, logical_block]` 仍然给出物理
block ID，页内 `token_map` 再告诉 kernel 该逻辑 token 是 anchor 还是 INT4，及其
在对应 payload 中的 ordinal。

### 4.2 冷 prefill 数据流

```text
BF16 Q/K/V
   |
   +--> BF16 FlashAttention 2 ------------------> attention output
   |
   +--> block-local importance selection
          |
          +--> BF16 anchor K/V
          +--> groupwise INT4 K/V + FP16 scales
                    |
                    +--> uint8 compact PagedAttention pages

recent prompt suffix ---------------------------> BF16 hot ring
```

页构建不会参与当前 cold prefill 的 attention 数值计算。因此 prefill 的 attention
核心与 FA2 基线相同，额外成本来自重要性选择、量化、pack、token map 写入和热窗口
写入。

### 4.3 Decode 数据流

```text
current BF16 Q/K/V
   |
   +--> boundary check --> optional periodic retirement/packing
   |
   +--> compact cold pages: INT4 tokens + BF16 anchors --+
   |                                                     |
   +--> request-local BF16 hot ring ---------------------+--> online softmax
   |                                                     |    and output
   +--> current K/V, normally fused into hot-ring store -+
```

Decode 不会先把全部 INT4 KV 反量化成 BF16 tensor。解包、scale、QK、softmax 和
PV 都在 Triton attention kernel 内按 tile 在线完成。

### 4.4 Prefix Cache 命中数据流

```text
shared compact prefix --> Triton partial output + natural-log LSE --+
                                                                   +--> stable LSE merge
private BF16 suffix ----> FA2 partial output + natural-log LSE -----+
```

生产路径不会重建完整 BF16 prefix。共享 compact page 只读；请求自己的最后 256
个 prompt token 重新计算并保留 BF16。

## 5. 参数与状态定义

下文使用：

| 符号 | 含义 | 默认值 |
| --- | --- | ---: |
| `B` | vLLM KV block size | 16 |
| `D` | attention head size | 128 |
| `G` | INT4 channel group size | 32 |
| `rho` | 每个 block 的 BF16 anchor 比例 | 0.0625 |
| `A` | `ceil(B * rho)`，每 block anchor 数 | 1 |
| `W` | BF16 hot window | 256 token |
| `R` | decode retirement interval | 64 token |
| `P` | 固定 prompt 长度 | 依请求而定 |
| `L` | 当前完整 sequence length | 随 decode 增长 |

当前已经提交为 compact page 的逻辑前缀长度为：

```text
initial = floor(max(P - W, 0) / B) * B
periodic = floor(max(L - P, 0) / R) * R
max_allowed = floor(max(L - W, 0) / B) * B
committed(L) = min(initial + periodic, max_allowed)
```

含义如下：

- prompt 完成时，最后 `W` 个 token 不量化；
- decode 每生成 `R` 个 token，一次退休一个完整 interval；
- 默认 `R=64`、`B=16`，所以一次退休并 pack 4 个 block；
- `max_allowed` 保证最近的 BF16 窗口不会提前被覆盖；
- ring 实际容量不是 256，而是 `W + R + B = 336`，额外空间避免周期退休和
  block 边界之间发生覆盖。

## 6. 紧凑 page 布局

### 6.1 通用布局公式

每个物理 block、每个 KV head 使用相同的固定 byte layout：

```text
BF16 anchor K
BF16 anchor V
packed INT4 K
packed INT4 V
FP16 K scales
FP16 V scales
uint16 token map
alignment padding
```

其中：

```text
A = min(B, ceil(B * rho))
Q = B - A
N_group = ceil(D / G)
packed_channels = ceil(D / 2)
```

一个 INT4 token、一个 KV head 的主要存储为：

```text
K codes            = ceil(D / 2) bytes
V codes            = ceil(D / 2) bytes
K scales           = N_group * 2 bytes
V scales           = N_group * 2 bytes
total               = 2 * ceil(D / 2) + 4 * N_group bytes
```

默认 `D=128, G=32` 时，一个 INT4 token 每 KV head 是 `64 + 64 + 8 + 8 =
144 bytes`。同一 token 的 BF16 K/V 是 `128 * 2 * 2 = 512 bytes`。

### 6.2 默认 16-token page 的精确布局

默认 `B=16, D=128, G=32, A=1, Q=15` 时：

| 区域 | byte offset | 大小/每 KV head |
| --- | ---: | ---: |
| anchor K，1 x 128 BF16 | 0 | 256 B |
| anchor V，1 x 128 BF16 | 256 | 256 B |
| 15 个 token 的 packed INT4 K | 512 | 960 B |
| 15 个 token 的 packed INT4 V | 1472 | 960 B |
| K scales，15 x 4 FP16 | 2432 | 120 B |
| alignment padding | 2552 | 8 B |
| V scales，15 x 4 FP16 | 2560 | 120 B |
| alignment padding | 2680 | 8 B |
| token map，16 x uint16 | 2688 | 32 B |
| **合计** | 0 | **2720 B** |

对应 BF16 K/V page 为：

```text
16 tokens * 128 channels * 2 bytes * (K + V) = 8192 bytes/head
```

所以仅看 page：

```text
8192 / 2720 = 3.0118x
```

allocator 看到的每逻辑 token、每 KV head 大小是 `2720 / 16 = 170 bytes`，
而不是用 BF16 page 大小伪装的逻辑量化。这是当前版本能真实增加 KV block 数的
关键。

### 6.3 INT4 量化方式

量化不是一个 block 共享一个 scale。每个非 anchor token、每个 KV head、每个
channel group 都单独计算，K 和 V 也使用不同 scale：

```text
scale = max(abs(group)) / 7
q = clamp(round(x / scale), -8, 7)
```

零 group 使用安全 scale 1。两个有符号 INT4 code 打包进一个 `uint8`，低 nibble
放偶数 channel，高 nibble 放奇数 channel。scale 以 FP16 保存。

`token_map` 是 `uint16`：

- 高位未置位时，值是 BF16 anchor ordinal；
- `0x8000` 高位置位时，低位是 INT4 token ordinal。

当前实现对一个 block 的所有 KV head 使用同一组 anchor token，因此映射规则、
PagedAttention 定位和 GQA/MQA 读取都保持规则。

## 7. 重要 token 如何选择

每个 block 选择固定 `A=ceil(B*rho)` 个 token。默认 `rho=0.0625` 和 `B=16`
时恰好是 1 个 BF16 token，即实际 anchor 比例为 6.25%。

对每个 KV head，属于它的 query heads 先求均值，然后计算 block 内每个 token 的
key 与参考 query 的点积绝对值；一个 token 对所有 KV heads 取最大分数，最后在
block 内做 top-A：

```text
q_ref[h_kv] = mean(Q_ref[query heads mapped to h_kv])
score[token] = max_h_kv(abs(dot(K[token, h_kv], q_ref[h_kv])))
anchor = top_A(score)
```

参考 query 的选择：

- prefill：每个 block 使用该 block 最后一个逻辑 token 的 query；
- decode retirement：使用触发当前 retirement 的 decode query。

这样一个 prompt block 的 anchor 只依赖该 block 已有的信息，不依赖未来 prompt
token，保证 chunked prefill 和 Prefix Cache 发布后的页面稳定。block 一旦 pack，
anchor 不再动态晋升或降级；当前实现没有原版 HyQuant 的全局运行列质量累计。

## 8. Prefill 实现

### 8.1 One-shot prefill

对没有 Prefix Cache 命中的完整 prompt：

1. 接收 BF16 Q/K/V；
2. 直接以 BF16 K/V 调用 `flash_attn_varlen_func(..., fa_version=2)`；
3. 对最后 `W` token 之前的完整 block 做重要性选择和紧凑 pack；
4. 把最近 token 写入 BF16 hot ring；
5. 返回 FA2 的 attention output。

因此，量化误差不会影响当前 prompt token 的 prefill attention 输出；它从后续
decode 读取这些 KV 时开始生效。没有 FA2 的诊断环境存在 PyTorch SDPA dense
fallback，但正式性能结果不使用它。

### 8.2 Chunked prefill

chunked prefill 的难点是后续 chunk 做 causal attention 时需要前面所有 prompt
K/V，同时 scheduler 可能重排请求。当前方案为每个活跃请求分配稳定
`state_slot`，并在 `_prefill_staging` 中维护尚未命中的完整 BF16 suffix：

1. 第一个 chunk 创建按完整 uncached suffix 容量分配的 BF16 K/V staging；
2. 后续 chunk 必须连续，按逻辑 offset 追加；
3. 当前 chunk 使用矩形 causal FA2，对 accumulated BF16 suffix 计算；
4. 每当本 chunk 新完成可共享的完整 block，就使用 block-final query 构建 compact
   page；
5. 最后 `W` 个 prompt token 不发布到 Prefix Cache；
6. prompt 最后一个 chunk 完成后释放 staging。

这里需要区分两个概念：compact page 可以随着 chunk 完成而提前写入，方便同一步
的 Prefix Cache reader；但 prefill attention 本身仍使用 BF16 staging，不读回
这些量化页。

该方案保持数值和 page 稳定性，但 staging 是 KV allocator 之外的临时 BF16
内存，而且每个 chunk 会再次对 accumulated suffix 做 attention。chunking 的目标
是接纳长 prompt、参与连续批处理和避免一次性超大 token batch，不保证比 one-shot
prefill 延迟更低。

### 8.3 Mixed prefill/decode batch

`HyQuantMetadataBuilder` 通过 vLLM 的
`split_decodes_and_prefills(..., treat_short_extends_as_decodes=False)` 分开 batch 前部
的单 token decode 与后部 prefill。每一部分使用自己的 block table、sequence
length、state slot 和 output slice，最后仍写入调用者提供的统一 output buffer。

Full CUDA Graph 会填充虚拟零长度 request。`backends/utils.py` 中额外修复了
`is_prefilling` 只覆盖真实 request、不能直接与 padded rows 整体广播的问题。

## 9. Decode 实现

### 9.1 为什么必须自定义 kernel

标准 FA2/PagedAttention 假设 K/V 行具有统一 BF16/FP16/FP8 表示。当前一个逻辑
序列同时包含：

- compact page 中的 packed INT4 K/V；
- compact page 中的 BF16 anchor；
- 单独 hot ring 中的 BF16 recent token；
- 当前 step 的 BF16 K/V。

如果先把冷区全部反量化成 BF16，再调用 FA2，会失去显存带宽收益并产生随上下文
线性增长的临时 tensor。因此当前 Triton kernel 直接遍历 block table，在寄存器/
tile 内解包并用 online softmax 合并所有来源。

### 9.2 默认快速路径：grouped INT8-QK

Qwen3-4B 的常用形状满足：

```text
block_size=16
head_size=128
group_size=32
anchor_count=1
query_heads / kv_heads=4
```

此时 `_run_int8_decode()` 直接调用低 ABI 的 grouped INT8-QK split-K kernel：

1. 当前 BF16 query 按 32-channel group 临时量化为 INT8，scale 使用
   `max(abs(q_group))/127`；
2. 冷区 K 保持 INT4 code，INT8 query 与解包后的 INT4 K 做整数 dot；
3. query scale 与持久化 K scale 恢复 QK 的实数尺度；
4. V 的 INT4 code 在 PV 累积时按自己的 group scale 在线恢复；
5. BF16 anchor 和 hot token 走 BF16 路径并进入同一个 online softmax；
6. context 按多个 split-K CTA 并行扫描，reducer 用 LSE 规则合并 partial state。

这里“INT8-QK”只描述 **query 与 quantized K 的临时计算方式**。持久 KV 格式没有
变成 INT8，仍然是 `hyquant_k4v4`，V 也仍为 INT4。

### 9.3 通用和兼容路径

不满足 Qwen3 快速形状时，dispatch 顺序为：

1. `hyquant_decode_groupdot.py`：优先的通用 packed group-dot/split-K；
2. `hyquant_decode_fast.py`：非 fused current-store 时的低 ABI BF16 group-dot；
3. `hyquant_decode_grouped.py`：兼容旧的 grouped materializing tile kernel；
4. `hyquant_block.py` 中的较通用 Triton mixed-page kernel；
5. 最后才是 PyTorch reference fallback，用于正确性和异常环境，不是性能路径。

这些路径都不应该在全局内存中构造完整 BF16 prefix。部分兼容 kernel 会在单个
tile/寄存器范围内恢复 BF16，这是计算实现细节，不等于全 cache materialization。

### 9.4 Split-K、workspace 与 dispatch

split 数、`BLOCK_KV`、`BLOCK_H`、warp 和 pipeline stage 根据 batch/context 的
静态 bucket 选择。默认 A800 策略中：

- B1、context 2K 到 8K 前使用 16 splits；
- B1、context 至少 8K 使用 64 splits；
- 更大 batch 根据已有 request 并行度使用较保守的 splits；
- MHA 或很短上下文可以不 split。

partial output 和 LSE 放在 vLLM workspace manager 提供的复用 FP32 buffer 中，
最大预留 64 splits。launch policy 在 metadata build 时计算一次，由所有 attention
layer 共用，避免每层重复解析环境变量和分配 workspace。

### 9.5 当前 token store 与 K/V stride

普通单 token decode 会把当前 K/V 写入 BF16 hot ring。生产快速路径可把这个写入
融合进 decode kernel，少一次 launch。retirement boundary 必须先读取旧 ring，再
发布 current row，顺序不能交换。

Qwen3 fused QKV 输出的 K view 与 V view 不保证 token stride 相同。早期多请求 bug
正是所有 kernel 错把 K stride 同时用于 V。当前所有 grouped 路径分别传递：

```text
stride_current_key_token / stride_current_key_head
stride_current_value_token / stride_current_value_head
```

这条约束对 batch 1 不明显，因为第 0 行 offset 为 0，但会直接破坏 batch 2 以上
的 V 地址，因此是多请求正确性的关键。

## 10. 周期量化与 CUDA Graph

### 10.1 为什么批量 retirement

如果每生成一个 token 都量化刚离开窗口的数据，每层每 step 都会增加选择、scale
计算和 pack launch。默认每 64 token 处理一次，且每次对 4 个 16-token block
一起完成，显著降低摊销开销。

GPU retirement kernel 读取 device-side `seq_lens`、`prompt_lens` 和稳定
`state_slots`，只在真实 boundary 激活；普通 step 是 device-side no-op。它在同一
packing launch 内重新计算确定性的 block-local anchor，随后写 token map、INT4
payload、scale 和 BF16 anchor，避免跨 kernel 同步。

### 10.2 Graph 路径

backend 声明支持 `UNIFORM_SINGLE_TOKEN_DECODE` CUDA Graph：

- 普通 one-token decode 可以 capture/replay；
- GPU graph retirement 可用时，boundary 操作也记录在 graph 内并由 device predicate
  控制；
- GPU retirement 被禁用或不可用时，runner 只让 retirement boundary step 走 eager，
  其他 decode step 保持 graph；
- host fallback 保留正确性，但不是目标性能路径。

`state_slot` 不能使用当前 batch row，因为 scheduler 会重排、抢占并复用 batch
位置。V2 runner 使用稳定 request state index；legacy runner 使用 request ID 到
空闲 slot 的映射，并在完成或 preemption 时回收。

## 11. Prefix Cache 实现

### 11.1 可共享范围

为避免共享页包含请求私有 BF16 tail，`AttentionSpec` 新增
`prefix_cache_private_tail_tokens`。HyQuant 把它设置为 `W`。对 prompt 长度 `P`：

```text
shareable_prefix = floor(max(P - W, 0) / B) * B
```

`KVCacheManager` 在 lookup、publish 和显式 cache 操作上都用这个上限。生成 token
永远不会被错误注册为共享 prompt prefix。

当前只支持完整物理 block 命中，所以：

```text
prefix_match_unit == block_size
```

默认 block size 16 时，两者都必须为 16。未显式提供 `prefix_match_unit` 时，
HyQuant 启动校验自动设为 block size。

### 11.2 Writer 与 reader

- cold writer 仍以 BF16 FA2 做 prefill；
- writer 对每个完成且可共享的 block 构建 compact page；
- backend 在运行 reader attention 之前先 pack 当前 batch 的 writer，支持同一步
  duplicate request 发布/命中；
- reader 只读 shared compact prefix，不重新选择 anchor、不原地 repack；
- reader 从 cached prefix 末尾到 prompt 末尾的 suffix 重新计算为 BF16，并建立
  自己的 hot ring。

### 11.3 第二阶段优化

Prefix Cache 第一阶段曾将 compact prefix 完整反量化成 BF16，再拼接 suffix 并
调用 FA2。它能工作，但每一层都产生大临时 tensor 和额外带宽。

当前第二阶段已经删除生产 backend 中的 materialize + concat 路径：

1. Triton 直接对 shared compact pages 计算 prefix partial output 和 natural-log
   LSE；
2. FA2 对 request-private BF16 suffix 计算 partial output 和 LSE；
3. vLLM `merge_attn_states` 用稳定 LSE 合并两部分；
4. compact kernel 不可用时，cache-hit 请求直接报错，不静默回到昂贵旧路径。

`materialize_hyquant_blocks()` 仍保留在 `hyquant_block.py`，但只作为 tensor test 和
离线 benchmark oracle；`hyquant_attn.py` 不导入、不调用它。

## 12. vLLM 源码改动总览

相对基线提交，当前 HyQuant 累计涉及 31 个源码/测试文件：19 个已有文件被修改，
12 个文件新增。下面按职责说明，而不是按历史 run 重复列举。

### 12.1 新增 backend 与 kernel

| 文件 | 作用与关键思路 |
| --- | --- |
| `vllm/v1/attention/backends/hyquant_attn.py` | HyQuant V1 backend 主入口；定义 backend 能力、metadata builder、page shape、prefill staging、FA2 prefill、direct-prefix + suffix merge、hot ring、periodic retirement、decode dispatch、workspace 和 JIT warmup。 |
| `vllm/v1/attention/ops/hyquant_common.py` | `hyquant_k4v4` 常量、默认参数、committed-length 状态机、固定 page ABI、offset/size 计算、groupwise symmetric INT4 pack/unpack 与 token map。 |
| `vllm/v1/attention/ops/hyquant_block.py` | PyTorch page packer、gather/materialize oracle、block-local token 选择、reference decode、通用 mixed-page Triton kernel，以及 grouped decode 的分层 dispatch。 |
| `vllm/v1/attention/ops/hyquant_hot.py` | 将 prefill/decode 的 BF16 K/V 写入 request-local circular hot ring；支持独立 K/V stride、prefill/decode 两种谓词和启动预热。 |
| `vllm/v1/attention/ops/hyquant_retire.py` | CUDA-graph-safe 周期退休；根据 device metadata 识别 boundary，在 GPU 上选择 anchor、做 groupwise INT4 并写 compact page。 |
| `vllm/v1/attention/ops/hyquant_decode_int8.py` | 默认 Qwen3-4B 形状的低寄存器 INT8-QK split-K decode；当前 token fused store；枚举所有默认可达 launch 配置并在初始化预热。 |
| `vllm/v1/attention/ops/hyquant_decode_groupdot.py` | 通用 packed group-dot decoder、静态 split policy、tile/warp/stage policy、split reducer 和 INT8-QK/fast decoder dispatch。 |
| `vllm/v1/attention/ops/hyquant_decode_fast.py` | 非 fused current-store 场景的低 ABI BF16 group-dot split-K kernel，减少通用 kernel 的参数和寄存器压力。 |
| `vllm/v1/attention/ops/hyquant_decode_grouped.py` | 当前 page ABI 的兼容 grouped decoder，覆盖 group-dot fast path 拒绝的合法布局，并作为 A/B/reference 级 fallback。 |
| `vllm/v1/attention/ops/hyquant_prefill.py` | Prefix Cache 命中的 direct compact-prefix attention；通用 tiled MHA/GQA/MQA 路径，输出 partial result 和自然对数 LSE，负责 warmup。 |
| `vllm/v1/attention/ops/hyquant_prefill_groupdot.py` | 针对 B16/D128/G32/A1 的 Prefix Cache 快速路径；一个 compact page tile 复用于多个 query token 和 GQA heads，不构造 dense prefix。 |

### 12.2 配置、注册与 allocator contract

| 文件 | 修改内容 |
| --- | --- |
| `vllm/config/cache.py` | 注册 `hyquant_k4v4`；加入 top ratio、group size、window 和 retirement interval；日志明确 compact mixed K4/V4 page。 |
| `vllm/engine/arg_utils.py` | 暴露四个 CLI/EngineArgs 参数；校验 block 对齐、Prefix Cache match unit，并在启动时拒绝 speculative、per-layer KV dtype skip、offload、connector 和 DCP。 |
| `vllm/platforms/cuda.py` | 把 `HYQUANT` 放入 CUDA attention backend 可选列表。 |
| `vllm/utils/torch_utils.py` | 把 `hyquant_k4v4` 映射为 `torch.uint8` backing storage，并标记为 quantized KV dtype。 |
| `vllm/v1/attention/backends/registry.py` | 注册 `AttentionBackendEnum.HYQUANT` 到 `HyQuantAttentionBackend`。 |
| `vllm/v1/kv_cache_interface.py` | 新增 `KVQuantMode.HYQUANT_K4V4`；为 `AttentionSpec` 增加 packed `state_content_bytes` 和 Prefix Cache private tail contract，并在 spec merge 中传播。 |
| `vllm/v1/core/kv_cache_manager.py` | 按 private BF16 tail 限制可查找、可发布的 prompt prefix；只共享完整对齐 block，并保护 eviction/refcount 语义。 |

关键点在 `HyQuantAttentionBackend.customize_spec()`：它把原始 attention spec 的
cache dtype 改为 `torch.uint8`，并把 `state_content_bytes` 设置为布局计算出的真实
`bytes_per_token`。`get_kv_cache_shape()` 返回：

```text
[num_blocks, num_kv_heads, head_page_bytes]
```

如果 layout 大小与 `AttentionSpec.page_size_bytes` 不一致，会导致容量虚报或越界，
所以这是整个适配中最先需要保证的 contract。

### 12.3 Attention metadata 与 runner

| 文件 | 修改内容 |
| --- | --- |
| `vllm/v1/attention/backend.py` | 在 `CommonAttentionMetadata` 增加 state slot、固定 prompt length、初始 cached-prefix length 及 CPU mirror，并在 unpadded/slice 时保留。 |
| `vllm/v1/attention/backends/utils.py` | virtual batch、KV-sharing metadata 和 decode/prefill split 时传播 HyQuant 字段；修复 CUDA Graph padded request 的 prefill flag 长度。 |
| `vllm/v1/worker/gpu/attn_utils.py` | 把 runner 准备的 HyQuant 字段送进每个 attention group 的 metadata builder。 |
| `vllm/v1/worker/gpu/input_batch.py` | 预分配 GPU/CPU metadata buffer，并把对应 slice 放入 `InputBatch`，避免 step 内反复分配。 |
| `vllm/v1/worker/gpu/model_runner.py` | V2 runner 根据稳定 request index 建立 slot/prompt/prefix metadata；计算 retirement boundary；GPU graph retirement 不可用时只让 boundary step eager。 |
| `vllm/v1/worker/gpu/model_states/default.py` | 从 `InputBatch` 把 HyQuant 字段转交给通用 attention metadata。 |
| `vllm/v1/worker/gpu/states.py` | 在 V2 `RequestState` 中保存请求首次进入 worker 时的 immutable cached-prefix length。 |
| `vllm/v1/worker/gpu_model_runner.py` | legacy runner 的 request ID 到 stable slot 映射、空闲 slot 回收、prompt/prefix metadata buffer 与构建。 |
| `vllm/v1/worker/ubatch_utils.py` | microbatch 切分时同步切 positions、prefill flag 和全部 HyQuant request-level 字段。 |

GPU 与 CPU mirror 同时存在是有意设计：kernel 使用 GPU tensor，而 prefill staging
检查、退休边界判断和 Python request 生命周期使用 CPU 值，避免每个 attention
layer 调用 `.item()` 导致 GPU 同步。

### 12.4 测试文件

| 文件 | 覆盖内容 |
| --- | --- |
| `tests/kernels/attention/test_hyquant.py` | compact page size、top ratio 0/1、groupwise round trip、token map、importance、retirement、FA2 rectangular prefill、chunk staging、direct Prefix Cache、LSE merge、MHA/GQA/MQA、fused store、split policy、JIT warmup 和多请求独立 K/V stride。 |
| `tests/v1/attention/test_attention_splitting.py` | HyQuant metadata 在 ubatch/unpadded 中完整保留，以及 Full CUDA Graph padded prefill flag。 |
| `tests/v1/core/test_kv_cache_utils.py` | `prefix_cache_private_tail_tokens` 在 attention spec merge 中保持。 |
| `tests/v1/core/test_prefix_caching.py` | private tail 的 publish/lookup 上限、generated block 不进入 prompt cache、eviction 与 refcount 回收。 |

## 13. 配置项与约束

| 参数 | 默认 | 含义 | 约束 |
| --- | ---: | --- | --- |
| `--attention-backend HYQUANT` | 无 | 显式选择自定义 backend | CUDA、SM80+ |
| `--kv-cache-dtype hyquant_k4v4` | `auto` | 使用紧凑 mixed K4/V4 page | 必须与 HYQUANT backend 配对 |
| `--hyquant-top-ratio` | 0.0625 | 每 block BF16 anchor 比例 | `[0, 1]`，数量向上取整 |
| `--hyquant-group-size` | 32 | 每个 INT4 scale 覆盖的 channel 数 | 正偶数 |
| `--hyquant-window-size` | 256 | 请求私有 BF16 最近窗口 | block 对齐，至少 16 |
| `--hyquant-retire-interval` | 64 | decode 批量退休周期 | block 对齐，至少 16 |
| `--block-size` | vLLM 默认 | PagedAttention 物理 block | backend 声明 16/32/64 |
| `--prefix-match-unit` | 自动 | Prefix Cache 匹配粒度 | 启用 APC 时必须等于 block size |

下列环境变量用于实验、诊断或内部 tuning，不应视为稳定公共 API：

| 环境变量 | 默认 | 用途 |
| --- | ---: | --- |
| `VLLM_HYQUANT_FUSE_CURRENT_STORE` | 1 | decode kernel 融合 current K/V hot-ring 写入 |
| `VLLM_HYQUANT_GPU_RETIRE` | 1 | 使用 Triton GPU retirement |
| `VLLM_HYQUANT_GRAPH_RETIRE` | 1 | 把 device-predicated retirement 放入 CUDA Graph |
| `VLLM_HYQUANT_DECODE_INT8_QK` | 1 | 启用默认形状 INT8-QK specialization |
| `VLLM_HYQUANT_GROUPED_STRICT` | 0 | kernel 失败时抛错而不是进入兼容 fallback |
| `VLLM_HYQUANT_DECODE_SPLITS` | 自动 | 覆盖静态 split count，仅用于 A/B |
| `VLLM_HYQUANT_DECODE_BLOCK_KV` | 自动 | 覆盖 KV tile |
| `VLLM_HYQUANT_DECODE_BLOCK_H` | 自动 | 覆盖 query-head tile |
| `VLLM_HYQUANT_DECODE_WARPS` | 4 | 覆盖 warp 数 |
| `VLLM_HYQUANT_DECODE_STAGES` | 自动 | 覆盖 software pipeline stage |

使用非默认 tuning override 可能触发未预热的 Triton specialization，也没有完成
与默认矩阵相同的性能/正确性验收。

## 14. 当前支持范围

### 14.1 已支持并有测试覆盖

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 标准 decoder self-attention | 支持 | 只支持 causal decoder attention |
| MHA / GQA / MQA | 支持 | tensor 级均覆盖；Qwen3-4B GQA 有模型级验证 |
| BF16 Q/K/V | 支持 | backend 当前只声明 `torch.bfloat16` |
| Compact K4/V4 page | 支持 | INT4 code + per-token/head/group FP16 scale |
| Block size 16/32/64 | 支持 | 通用 kernel 有覆盖；性能主路径验证 B16 |
| One-shot prefill | 支持 | 冷请求 BF16 FA2 |
| Chunked prefill | 支持 | 稳定 slot + BF16 suffix staging |
| Pure decode | 支持 | 每请求每 step 一个 decode token |
| Mixed prefill/decode | 支持 | metadata split 后分别执行 |
| 多请求 grouped decode | 支持 | 已修复并覆盖独立 K/V stride |
| 周期 retirement | 支持 | 默认每 64 token 批量 pack 4 blocks |
| 本地 Prefix Cache | 支持 | 同 engine、同进程、完整物理 block hit |
| 同一步 writer/reader | 支持 | writer page 在 reader attention 前 pack |
| V2 GPU runner | 支持 | 当前 Qwen3-4B 主验证路径 |
| Legacy GPU runner | 支持 | 小模型 smoke 与共享 backend 路径 |
| Ordinary decode CUDA Graph | 支持 | uniform single-token decode |

接口允许 `head_size >= 16` 且为 8 的倍数；direct-prefix 通用 Triton 路径要求偶数
head size、不超过 256。完成最充分模型级和性能级验证的组合仍是
`B16/D128/G32/A1/GQA ratio 4`。

### 14.2 明确不支持

- speculative decoding；
- per-layer KV cache dtype skipping；
- KV offload；
- KV connector、KV transfer；
- 外部或跨进程 Prefix Cache；
- partial-block Prefix Cache hit；
- DCP 和 PCP；
- MLA；
- 模型自带 sliding-window attention；
- ALiBi；
- encoder-decoder cross-attention；
- non-causal attention；
- multimodal prefix 特殊路径；
- SinkAttention sparse mask；
- logits soft cap。

这些组合应在 backend capability 或 engine 初始化时拒绝，而不是静默换成另一个
attention backend。普通 TP/PP 没有在当前结果中完成系统验收；已有模型数字均为
TP=1，不能据此宣称多卡性能或完整兼容性。

## 15. 正确性与准确率验证

### 15.1 最新回归结果

v51 在物理 GPU1/A800 上完成：

| 测试 | 结果 |
| --- | ---: |
| `tests/kernels/attention/test_hyquant.py` | 52 passed |
| attention splitting + KV utility + Prefix Cache | 198 passed |
| fused INT8-QK fresh-cache strict JIT matrix | 10 个默认配置通过 |
| non-fused INT8-QK fresh-cache strict JIT matrix | 10 个默认配置通过 |
| Qwen3-4B P4096/O32 strict inference JIT monitor | 通过 |

Prefix Cache v50 还验证了 direct compact output/LSE、MHA/GQA/MQA、不同 prefix
长度、zero-hit row、逻辑 block table 顺序、shared-page immutability，以及 compact
prefix 与 BF16 suffix 的 merge。

Chunked prefill v48 验证了：

- Qwen3-4B V2，prompt 128/513，chunk size 256 与 one-shot 的 16 个生成 token
  完全一致；
- llama-68m legacy runner，chunk size 64 smoke 通过；
- 矩形 FA2 prefill 与 dense causal oracle BF16 exact equality；
- chunk 重排、非对齐 chunk、slot 复用和 Prefix Cache suffix staging。

### 15.2 多请求 bug 与修复后准确率

v46 的 200 条 LongBench/GSM8K/MATH-500 run 曾出现严重崩溃，但根因不是已经确认的
INT4 精度损失，而是 grouped decode 对 V 使用了错误的 K token stride。该结果只能
作为 **修复前失败证据**，不能代表当前代码。

v47 修复后，用同一组 8 道 GSM8K 做回归：

| 配置 | 准确率 | 最终答案与 FA2 一致 |
| --- | ---: | ---: |
| HyQuant batch 1 | 87.5% | 8/8 |
| HyQuant batch 2 | 87.5% | 8/8 |
| HyQuant batch 8 | 87.5% | 8/8 |

这只是小规模 smoke test，不是完整统计准确率结论。修复后的 200 条正式套件尚未
重跑，因此当前能合理表述为“未在小规模回归中观察到准确率下降，并修复了已知
多请求地址错误”，不能表述为“已证明与 BF16 全面等精度”。

量化 replay 与 cold BF16 writer 也不要求 token bitwise identical。v50 P4096/O32
Prefix Cache 测试中，replay 与 cold output 的位置一致率为 93.75%；shared page
字节保持不变，差异属于预期量化漂移。

详细记录：

- [`v47 多请求修复`](../vllm_experiments/runs/20260914-hyquant-multirequest-fix-v47/ACCURACY.md)
- [`v50 Prefix Cache 测试`](../vllm_experiments/runs/20260914-hyquant-prefix-cache-v50/TESTS.md)
- [`v51 Decode/JIT 测试`](../vllm_experiments/runs/20260914-hyquant-decode-jit-opt-v51/TESTS.md)

## 16. 显存与 KV 容量效果

### 16.1 理论结果

默认页是 2720 B/head，BF16 页是 8192 B/head，纯 page footprint 理论为
`3.0118x` 容量。实际引擎还包含：

- 每层、每活跃 request 的 BF16 hot K/V ring；
- prefill staging；
- split-K workspace；
- anchor/retirement scratch；
- 模型权重、激活和 CUDA Graph pool；
- allocator 对齐及运行时保留空间。

因此真实总容量会略低于理论 page 比例。hot ring 的存储量随
`num_layers * max_num_seqs * num_kv_heads * (W + R + B) * D` 增长，配置很大的
`max_num_seqs` 时必须纳入预算。

### 16.2 实测结果

Qwen3-4B、相同 `gpu_memory_utilization=0.9` 的 v46 启动日志：

| KV backend | vLLM 报告 KV token slots |
| --- | ---: |
| FA2 + BF16 KV | 428,240 |
| HyQuant compact K4/V4 | 1,222,992 |
| **容量比** | **2.86x** |

另一组相同配置对照记录为 318,304 对 957,184，约 `3.0x`。在
`gpu_memory_utilization=0.45` 的 chunked-prefill run 中，FA2 约 201K，HyQuant
约 605K 到 606K，同样约 `3.0x`。

这意味着相同 allocator 预算下，可以容纳大约三倍的持久 KV token，从而提高长
上下文或大量并发请求的 admission 空间、降低因 KV 不足触发 preemption/OOM 的
概率。它不等于所有 workload 都能直接实现三倍 request throughput，因为 hot
ring、临时 prefill 内存、模型算力和 scheduler 仍是独立约束。

## 17. 当前性能效果

### 17.1 如何解释“加速”

当前有三类不同的倍率，不能混为一谈：

1. **相对旧 HyQuant 实现的优化倍率**；
2. **HyQuant 单个 attention 核心相对 FA2**；
3. **完整模型端到端相对 FA2**。

当前代码没有证据支持“HyQuant 在所有 batch/length 上端到端快于 FA2”。稳定、
可复现的主收益是 KV 容量。

### 17.2 最新 decode 单层结果

v51 的 Qwen3-4B shape、B1、A800 中位数：

| Context | HyQuant INT8-QK + reducer | HyQuant backend | FA2 |
| ---: | ---: | ---: | ---: |
| 4,096 | 0.09114 ms | 0.10445 ms | 0.07680 ms |
| 31,832 | 0.14234 ms | 0.15565 ms | 0.14438 ms |

在 31.8K：

- HyQuant 核心 kernel + reducer 比 FA2 约快 1.4%，约 `1.014x`；
- 加上 backend wrapper 后为 0.15565 ms，反而比 FA2 高约 7.8%；
- 说明 INT4 带宽收益已经能抵消在线解包计算，但 Python/dispatch、store、workspace
  和边界处理仍有固定开销。

在 4K，FA2 仍明显更快，说明低 batch、短/中 context 下 split/reducer 和 launch
固定成本还不能被较少的 KV bytes 抵消。

v51 把 B1、context 至少 8K 的 split 数从 32 调到 64。P8192/O64 的真实
Qwen3-4B HyQuant A/B：

| 指标 | 旧 HyQuant | 当前 HyQuant | 改善 |
| --- | ---: | ---: | ---: |
| backend 单层 | 0.10342 ms | 0.10138 ms | 1.98% |
| mean TPOT | 19.834 ms | 19.335 ms | 2.52% |
| output throughput | 48.57 tok/s | 49.75 tok/s | 2.44% |

这张表是 **当前 HyQuant 相对旧 split policy** 的加速，不是相对 FA2 的 2.44%
加速。相同单层 run 的 FA2 为 0.07987 ms。

### 17.3 Prefix Cache 第二阶段

Qwen3-4B、B1、P4096/O32、chunk 512、两次 replay：

| 路径 | Replay TTFT | Replay wall | TPOT | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| FA2 BF16 APC | 39.42 ms | 0.6043 s | 18.23 ms | 52.95 |
| HyQuant v49 materialize | 76.28 ms | 0.6605 s | 18.85 ms | 48.45 |
| HyQuant 当前 direct compact | 48.18 ms | 0.6360 s | 18.96 ms | 50.32 |

当前 Prefix Cache 相对 v49：

- replay TTFT 降低 36.8%，即约 `1.58x`；
- replay wall 降低 3.7%；
- output throughput 提升 3.8%；
- 单层 cache-hit path 从 0.957 ms 降到 0.678 ms，即 `1.41x`；
- 单层临时 PyTorch allocation 从 114.06 MiB 降到 6.96 MiB，即减少 16.4 倍。

相对 FA2 APC，当前 HyQuant output throughput 为约 `0.95x`，wall 高 5.2%，TPOT
高 4.1%，TTFT 高 22.2%。差距的一部分来自 HyQuant 刻意少命中最后 256-token
BF16 private suffix：该对照中 FA2 命中 4080 token，HyQuant 命中 3840 token。

### 17.4 Prefill 开销

在早期但布局一致的 B1/31.8K 单层拆分中：

| 操作 | 中位数 |
| --- | ---: |
| FA2 prefill | 43.1165 ms |
| HyQuant 内部 FA2 call | 43.1923 ms |
| token selection | 0.3604 ms |
| groupwise K4/V4 pack | 5.4702 ms |
| BF16 hot store | 0.2888 ms |

FA2 核心差异只有约 0.18%，额外 prefill 工作主要是 page construction。该次完整
endpoint TTFT 约 3.03 s，二者接近；这些是单层诊断，不应直接乘层数推导端到端。

详细原始记录：

- [`v50 Prefix Cache 性能`](../vllm_experiments/runs/20260914-hyquant-prefix-cache-v50/PERF.md)
- [`v51 Decode/JIT 性能`](../vllm_experiments/runs/20260914-hyquant-decode-jit-opt-v51/PERF.md)
- [`32K Prefill/Decode 早期拆分`](../vllm_experiments/runs/20260913-current-hyquant-b1-32k/PERF.md)

## 18. 遇到的主要困难与演进

### 18.1 固定 PagedAttention page 与混合 token 精度冲突

最初容易想到“一个 page 全 BF16 或全 INT4”，但重要 token 通常散落在 block
内部；另一种早期方案虽然逻辑上写入 INT4，却仍让 allocator 按 BF16 page 保留
显存，只有格式没有容量收益。

最终方案是一个 block 内固定 `A` 个 BF16 token、`B-A` 个 INT4 token，所有区域
和 token map 都具有确定 offset。v23 开始 allocator 使用真实 byte footprint，并
删除了旧 `block_types`/分离 type-table 布局。历史实现只保留在
`vllm_experiments`，不再作为可导入源码存在。

### 18.2 “以 block 处理”不等于“一个 block 一个 scale”

block 只是 PageAttention 的分配、发布和退休单位。如果整个 block 共用 scale，
异常 channel 会压缩其他值的有效量化范围，精度风险很高。最终保持
per-token/per-head/per-group scale，只把调度和 kernel tile 对齐到 block。

### 18.3 Prefill 不应该改成低精度主计算

早期自定义 compact-domain prefill 既慢，又让 TTFT 和量化误差难以归因。当前将
cold prefill 明确恢复为 BF16 FA2，量化只作为 FA2 后的 page construction。这样
Prefill 主核与基线一致，也符合“主要优化 decode KV”的目标。

### 18.4 Chunked prefill 缺少跨 step 状态

单次 prefill 可以直接看到完整 prompt；chunked prefill 看不到之前 chunk 的
BF16 K/V，而且 batch row 会变化。为此必须把 request-level stable slot、固定
prompt length、cached-prefix length 同时贯穿 scheduler/runner/metadata/backend，
并在 request 完成、preempt 或 slot 复用时清理 staging。

### 18.5 第一代 decode 的不规则访问压倒了带宽收益

早期 kernel 每个 query head 串行扫描 block，逐 token 读 map、分支、解 nibble 和
scale。31.8K 单层曾测得 4.7759 ms，而 FA2 只有 0.1485 ms，导致端到端 TPOT
约慢 9.5 倍。后续依次引入：

- grouped GQA head 复用；
- packed group-dot，不先物化 BF16 tile；
- split-K 并行与 LSE reducer；
- coalesced block-table/scale load；
- INT8 query 与 INT4 K integer dot；
- current-row store fusion；
- GPU retirement；
- metadata policy 和 workspace 跨 layer 复用。

最终把同类 31.8K attention 核心降到约 0.142 ms，数量级问题基本消除，但
backend 固定开销和短 context 效率仍存在。

### 18.6 多请求 stride bug

K/V 来自 fused QKV projection 时可以拥有不同 token stride。共用 K stride 的代码
在 batch 1 看似正确，但 batch 2 以后读取错误 V 地址。这个问题导致 v46 形式上
看起来像“量化准确率崩溃”。v47 增加了不同 K/V stride、反向 state slots、反向
physical block table 和三条 grouped 路径的回归测试，修复后 8/8 最终答案恢复。

### 18.7 CUDA Graph 与周期写页

host Python retirement 不能出现在 CUDA Graph replay 中。完全禁用 graph 会让所有
普通 decode step 付出性能代价。当前 GPU kernel 用 device metadata 判断 active
boundary；若该能力关闭，则 runner 只让真正 boundary eager。Full graph 的 padded
request 又引出 `is_prefilling` 长度不一致问题，也已在 metadata split 中处理。

### 18.8 Prefix Cache 的共享/私有边界

如果把最后的 BF16 window 也发布为共享 compact page，reader 会缺失自己的热区
状态；如果 reader 原地 repack shared page，又会破坏其他请求。因此 scheduler
必须只 hash/publish `P-W` 之前的完整 block，reader 只读 compact prefix，BF16
tail 私有重算。

第一阶段 materialize 整个 prefix 虽然正确，却在单层产生 114 MiB 临时分配。
第二阶段改为 partial output + LSE merge，并在验证后删除 backend 中的旧生产
调用，避免两套实现长期并存。

### 18.9 Triton JIT 不能出现在首个真实请求中

split count、tile、warp、stage、fused ABI 和 tensor stride 都会产生 specialization。
普通 vLLM warmup 不一定覆盖 Prefix Cache LSE stride 或所有 decode bucket。当前：

- 删除了只影响 grid、不影响 kernel code 的 `batch_size constexpr`；
- 按 `max_model_len` 和 `max_num_seqs` 枚举所有默认可达配置；
- fused/non-fused ABI 只预热 engine 实际选择的一套；
- hot store、retirement、prefix 和 split reducer 都有 setup warmup；
- 从空 Triton cache 的严格监控证明首个 inference 不再触发默认路径 JIT。

完全新鲜的 Triton cache 仍会在 **engine startup** 编译；这里消除的是 inference
期间 JIT，不是让编译过程凭空消失。

### 18.10 不是所有看似并行的优化都有效

- CUDA Graph runtime dynamic split 会增加分支、除法和 reducer 浪费，A/B 后删除；
- tiled/parallel reducer 比 serial reducer 慢 24%-28%，没有进入生产源码；
- 过度泛化 Triton `do_not_specialize` 曾使 Prefix Cache kernel 从约 0.65 ms 退化到
  0.87 ms，最终只对真正变化的 LSE stride 做定向 warmup；
- 短上下文专用 decode kernel 按需求未继续扩展。

遵循当前项目的替换规则：新实现通过正确性和性能门槛后，旧生产路径要从 vLLM
源码删除；历史 diff、命令和数据只留在 `/home/zy/xbx/vllm_experiments`。

## 19. 当前已知限制与剩余瓶颈

1. **短/中上下文 decode**：INT4 unpack、split/reducer 和 launch 固定成本高于节省的
   KV bandwidth，FA2 通常更快。
2. **Backend wrapper**：长上下文核心已接近 FA2，但 metadata/dispatch、hot-store、
   workspace 和状态处理仍让完整 backend 略慢。
3. **Prefill pack**：冷 prefill attention 是 FA2，但每层仍需重要性选择与 K4/V4
   page construction。
4. **Chunk staging**：长 prompt 的 uncached suffix 暂存完整 BF16 K/V，位于 allocator
   预算之外；极端并发长 prompt 仍可能产生瞬时显存压力。
5. **BF16 hot ring**：按 `max_num_seqs` 预分配并存在于每层，增大 window 或并发上限
   会明显增加固定显存。
6. **Prefix Cache private suffix**：HyQuant 主动少命中最后 W token，TTFT 不会与
   FA2 的最大 block hit 完全相同。
7. **优化形状集中**：最高性能 specialization 针对 A800、B16/D128/G32/A1、GQA
   ratio 4；其他合法布局主要依赖通用 kernel。
8. **准确率样本有限**：已知多请求 bug 已修复，但修复后的完整 LongBench + math
   200 条套件未重跑。
9. **单卡验证为主**：DCP/PCP 明确不支持，普通 TP/PP 也尚无本轮完整验收结果。

## 20. 如果从原始 vLLM 重新实现，应按什么顺序修改

### 第一步：定义 dtype 与固定 page ABI

先修改 `cache.py`、`torch_utils.py` 和 `kv_cache_interface.py`，让 allocator 真正按
compact byte 数分配。优先写 page size/offset 单测；如果这一步不正确，后续 kernel
即使数值正确也可能越界或没有显存收益。

### 第二步：注册独立 attention backend

在 registry 和 CUDA platform 中注册 `HYQUANT`，backend 明确声明 BF16、decoder、
SM80+、block size 和不支持特性。不要冒用已有 INT4/TurboQuant dtype，否则 cache
语义和 benchmark backend 选择会变得不可验证。

### 第三步：先完成 page codec 与 PyTorch oracle

实现 layout、INT4 pack/unpack、token map、importance selector、page gather 和 dense
decode oracle。覆盖 top ratio 0、1 和默认混合页，再写 Triton。oracle 保留用于
测试，但不能成为生产长上下文路径。

### 第四步：接通 request-level metadata

把 state slot、prompt length、cached prefix length 同时接入 V2/legacy runner、
CPU/GPU buffer、common metadata、ubatch slice 和 request cleanup。重点验证 scheduler
重排、preemption、padding 和 slot reuse。

### 第五步：保持 cold prefill 为 BF16 FA2

先用 FA2 得到 attention output，再 pack 旧完整 block，并写 hot ring。chunked
prefill 需要 BF16 staging 和 block-final stable selection；不可用最后整个 prompt
query 重选早期 block，否则同一 page 会随 chunk 边界变化，Prefix Cache 也不稳定。

### 第六步：实现 direct mixed decode

kernel 必须直接消费 block table 和 compact page，在线合并 INT4、anchor、hot/current
KV；先实现通用正确路径，再以真实模型 shape 做 grouped/split-K/INT8-QK 优化。
K/V stride 必须独立，GQA 不应显式 repeat K/V。

### 第七步：加入周期 retirement 与 graph 语义

先用 committed-length 公式定义唯一状态机，再让 CPU 判断、GPU kernel 和测试共享
同一规则。保证 current store 与 retirement 的先后顺序；只有 GPU device-predicated
路径通过后才宣称 boundary graph-safe。

### 第八步：最后接 Prefix Cache

先在 scheduler 层定义 private tail 和 full-block publish 上限，再处理 reader。正确
实现应直接从 compact prefix 得到 output/LSE，与 BF16 suffix 的 FA2 state 合并；
shared pages 必须只读。

### 第九步：按层次验收

```text
layout/pack unit tests
    -> tensor attention vs BF16/dequantized oracle
    -> multi-request and lifecycle tests
    -> model greedy regression
    -> accuracy smoke/full set
    -> single-operator profile
    -> end-to-end FA2 comparison
    -> KV capacity and OOM/preemption pressure
```

性能结果必须确认日志实际选择 `AttentionBackendEnum.HYQUANT`，基线必须确认
`AttentionBackendEnum.FLASH_ATTN` 且 `fa_version=2`。不能把 fallback 或历史不同
环境的数据混入同一张正式表。

## 21. 阅读源码的建议顺序

为了最快理解当前实现，建议按以下顺序：

1. `vllm/v1/attention/ops/hyquant_common.py`：先看 page ABI 和状态公式；
2. `vllm/v1/attention/backends/hyquant_attn.py`：看 backend、metadata 和完整数据流；
3. `vllm/v1/attention/ops/hyquant_block.py`：看 pack、token map 与 dispatch；
4. `vllm/v1/attention/ops/hyquant_decode_int8.py`：看默认生产 decode；
5. `vllm/v1/attention/ops/hyquant_decode_groupdot.py`：看通用 decode 和 split policy；
6. `vllm/v1/attention/ops/hyquant_hot.py` 与 `hyquant_retire.py`：看周期窗口；
7. `vllm/v1/attention/ops/hyquant_prefill.py` 与
   `hyquant_prefill_groupdot.py`：看 Prefix Cache direct-prefix；
8. `vllm/v1/core/kv_cache_manager.py`：看 private tail 的共享边界；
9. V2/legacy runner 文件：看 request state 如何贯穿调度；
10. `tests/kernels/attention/test_hyquant.py`：用测试理解各 contract 和失败案例。

## 22. 实验记录位置

源码仓库只保留当前活跃实现。规划、旧实现 diff、原始 JSON、命令、日志、支持矩阵
和性能结论统一保存在：

```text
/home/zy/xbx/vllm_experiments
```

与当前功能最相关的最终记录：

- `runs/20260914-hyquant-multirequest-fix-v47`：多请求 stride 修复；
- `runs/20260914-hyquant-chunked-prefill-v48`：chunked prefill；
- `runs/20260914-hyquant-prefix-cache-v49`：Prefix Cache 第一阶段；
- `runs/20260914-hyquant-prefix-cache-v50`：direct compact Prefix Cache；
- `runs/20260914-hyquant-decode-jit-opt-v51`：decode JIT 消除和 B1/8K split 优化；
- `runs/20260914-hyquant-documentation-v52`：本文档变更记录。

最需要保留的最终判断是：**当前适配已经把 mixed K4/V4 KV 做成真实的 vLLM
PagedAttention 紧凑存储，也接通了主要单机 serving 生命周期；约三倍 KV 容量是
明确收益。性能在长上下文核心上已接近 FA2，但完整 backend 和短上下文还不能被
描述为普遍加速，准确率结论也仍应限定在已有的小规模修复后验证范围内。**
