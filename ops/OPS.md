# 运行说明（Qwen3.8-Flash-Next on CMP 170HX / WSL2）

## 当前状态

- **性能关键设置：`VLLM_WSL2_ENABLE_PIN_MEMORY=1`**（已写进 `run_container.sh`，默认值就是 1）。
  没有它性能会掉 30~35%，原因见文末「WSL2 pinned memory 门控」。
- 容器：`hong-pc`（镜像 `18gogogo/170hx1-qwen38nextf:sm80`），端口 **9393 → 8000**
- API：`http://localhost:9393/v1`，模型名 `Qwen3.8-Flash-Next`
- 模型：`~/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`（13 个分片，142.50 GiB，已 sha256 校验）
- 使用 GPU 0（`CUDA_VISIBLE_DEVICES=0`），`QWEN_CONTEXT=262144` 等参数按 18gogogo 的推荐配置

## 常用命令

```bash
docker logs -f hong-pc            # 跟踪日志（模型加载/请求日志）
curl -f http://127.0.0.1:9393/health
docker stop hong-pc && docker rm hong-pc          # 停止/删除
bash ~/vllm/run_container.sh --now                # 重新启动（跳过下载等待）
```

调整参数（改后重建容器）：

```bash
QWEN_GPU_MEMORY=0.90 QWEN_SEQS=8 bash ~/vllm/run_container.sh --now
CUDA_VISIBLE_DEVICES=1 bash ~/vllm/run_container.sh --now   # 换到 GPU 1
CONTAINER_NAME=hong-pc2 bash ~/vllm/run_container.sh --now  # 换容器名（或 --name hong-pc2）
bash ~/vllm/run_container.sh --help                         # 参数/环境变量一览
```

`run_container.sh` 支持的环境变量：`QWEN_CONTEXT`、`QWEN_SEQS`、`QWEN_BATCH_TOKENS`、
`QWEN_GPU_MEMORY`、`QWEN_MTP`、`CUDA_VISIBLE_DEVICES`、`PORT`（宿主端口）、
`VLLM_WSL2_ENABLE_PIN_MEMORY`（默认 1，**不要关**）、`PROFILE=1`（见下）、
`CONTAINER_NAME`（容器名，默认 `hong-pc`）、
`PERSIST_TRITON_CACHE`（默认 1，把 Triton 内核缓存挂到宿主 `~/vllm/triton_cache`）、
`TRITON_CACHE_HOST_DIR`（改缓存落地目录）。

容器名**只认** `CONTAINER_NAME` / `--name`，不再读通用的 `NAME`——本机 shell 会注入
`NAME=Code`（VS Code/pi 扩展设的），以前会让脚本去重建一个叫 `Code` 的容器。旧习惯
`NAME=xxx bash run_container.sh` 会被忽略并在启动时打印一条提示。

### Triton 缓存为什么要挂到宿主

内核（Triton）缓存原本只在容器内的 `/root/.triton/cache`，而 `run_container.sh` 每次
都会 `docker rm -f` 重建容器 → 缓存全丢，下一次请求用到没编译过的 shape 就要**在推理
中途冷编译**，引擎循环会卡在那个 step 上，`docker logs` 好几分钟一行不出（`/health`
仍 200，容易误判成死机）。2026-09-28 实例：模型默认 `temperature=1.0/top_k=20/top_p=0.95`
触发 `_gumbel_sample_kernel` 冷编译，`EngineCore` 阻塞约 **7.5 分钟**，日志里留下
`WARNING [jit_monitor.py:140] Triton kernel JIT compilation during inference`。
现在缓存挂在 `~/vllm/triton_cache`（约 94 MiB，容器以 root 写入，属主是 root 属正常），
只有**第一次**（或换镜像/改 kernel 后）需要付编译代价。关掉用 `PERSIST_TRITON_CACHE=0`。

## 性能结果（2026-09-27，3 轮中位数，128 输出 token）

| 并发 | 聚合 tok/s | 参考机 | 单请求 decode tok/s | 参考机 | TTFT 本机/参考 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | **107.03** | 103.30 | **112.27** | 110.99 | 0.067 / 0.105 |
| 4 | **282.53** | 277.22 | **79.14** | 77.95 | 0.154 / 0.195 |
| 8 | **472.99** | 428.97 | **65.19** | 60.63 | 0.165 / 0.183 |
| 16 | **754.31** | 667.13 | **52.97** | 47.32 | 0.211 / 0.262 |

Prefill tok/s（512/2048/8192/32768）：**2465 / 3094 / 3769 / 3122**（参考机 1616 / 1915 / 2636 / 2132）。
四个并发档位全部超过参考机。

修复前的对照（同一脚本，`measurements/chat.json`）：聚合 80.07 / 237.09 / 386.53 / 633.36。

### 排障提示

- **「服务超时/日志不动了」先看 `jit_monitor` 警告**：`docker logs hong-pc | grep jit_monitor`。
  若是 Triton 冷编译阻塞，等它跑完（可同时看 `docker exec hong-pc ls /root/.triton/cache | wc -l`
  是否还在涨）；根治办法是上面的宿主缓存 + 启动后先跑几条覆盖默认采样参数
  （`temperature=1.0/top_k=20/top_p=0.95`）的 warmup 请求。
- 判断「是引擎卡住还是 API 挂了」：`curl :9393/health` 200 只说明 API server 活着；
  真正看引擎要看 `curl :9393/metrics | grep num_requests_running` 和
  `engine 000: Avg ... throughput` 那行是否还在每 10s 出现。

- 仓库自带的 `bench_server.py`（`median_decode_tokens_s`）统计的是**真实 token 速率**，
  与参考表可直接比。`~/vllm/decode_probe.py` 打印的是 **SSE chunk 速率**；因为 MTP=1 时
  每个 chunk 平均带 1.76 个 token，所以它的数字要 ×1.76 才是 tok/s（例如 62.6 chunk/s ≈ 108 tok/s）。
- 6/6 smoke 里 `python` 一项**本身不稳定**（模型/量化问题，仓库 RESULTS.md 也提到）：
  同一 prompt 连跑 8 次得 `30,30,30,20,20,20,30,30`。这不是性能问题的征兆。

## 性能剖析（内核级）

`PROFILE=1 bash ~/vllm/run_container.sh --now` 会打开 torch profiler 端点：`PROFILE=1` 时
容器改用 `~/vllm/enable_profiler_entrypoint.sh` 作为 entrypoint，它在运行时给镜像自带的
`/opt/entrypoint.sh` 添加由 `/prof/enable_profiler` 标记文件门控的 `--profiler-config`，
然后 exec 原脚本（幂等，已在新容器中验证）。

```bash
PROFILE=1 bash ~/vllm/run_container.sh --now
curl -X POST http://127.0.0.1:9393/start_profile
python3 ~/vllm/decode_probe.py http://127.0.0.1:9393 128
curl -X POST http://127.0.0.1:9393/stop_profile   # trace 落到 ~/vllm/prof/*.pt.trace.json.gz
```

关闭：`rm ~/vllm/prof/enable_profiler && docker restart hong-pc`。
（现有这个容器是在 profiler 打开状态下启动的，重启后才会按标记文件关闭。）

**注意：本机 GPU 上 CUPTI 不可用**（`CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`），
所以 trace 里**没有 CUDA kernel 事件**，只有 CPU 事件与 `execute_context_*_generation_*` 注解。
要看 GPU 侧就看 `nvidia-smi` 的利用率/功耗采样，以及容器内微基准。

其他剖析工具：`~/dlvenv/bin/py-spy`（`docker run --rm --pid=container:hong-pc --cap-add=SYS_PTRACE ...`，
必须加 `--nonblocking -r 50`），`~/vllm/analyze_trace.py`（kineto trace 汇总）。

## 基准脚本（仓库自带）

`bench_prefill.py` 的地址硬编码为 `127.0.0.1:8000`，因此本地起了转发进程
`~/vllm/forward8000.py`（WSL 内 8000 → 9393，只影响 WSL 命名空间，不影响 Windows 的 8000）。

```bash
~/dlvenv/bin/python <repo>/benchmarks/verify_server.py --url http://127.0.0.1:9393 \
    --long --tools --output ~/vllm/measurements/smoke.json
~/dlvenv/bin/python <repo>/benchmarks/bench_server.py --label local \
    --output ~/vllm/measurements/chat.json --concurrency 1 4 8 16 --rounds 3
~/dlvenv/bin/python <repo>/benchmarks/bench_prefill.py --label local \
    --output ~/vllm/measurements/prefill.json --seed 729156 --repeats 3 --lengths 512 2048 8192 32768
```

## WSL2 pinned memory 门控（decode 变慢的根因，已修复）

**现象**：单请求 decode 只有 83 tok/s，参考机 111 tok/s；GPU 利用率 64~68%、功耗 133~150 W
（未打满 250 W），说明 GPU 在等而不是算不动。

**定位过程**：

1. `py-spy record -p <EngineCore pid>` → 主线程 25 s 里 21.7 s 在 CPU 上，其中 **76% 是
   `vllm/v1/worker/gpu/buffer_utils.py:58` 的 `uva`**（调用栈
   `execute_model → apply_staged_writes → copy_to_uva → copy_to_uva → uva`）。
2. `buffer_utils.py:58` 正是 `NonUvaBuffer.uva`，它每次调用都执行
   `self._uva[:n].copy_(self.cpu[:n])`，即**一次同步的 pageable→GPU 拷贝**。
3. 为什么走了 fallback：`platforms/cuda.py:306` `is_pin_memory_available()` 里
   WSL2 分支返回 `envs.VLLM_WSL2_ENABLE_PIN_MEMORY`，其**默认值是 0（False）**；
   于是 `is_uva_available()` 为 False，`UvaBufferPool` 全部退化成 `NonUvaBuffer`。
   本机内核 6.18.33.2-microsoft-standard-WSL2 ≥ 4.19.121，其实完全支持 pinned memory（实测可创建，
   且 `torch.ops._C.get_cuda_view_from_cpu_tensor` 零拷贝映射在 WSL2 下也能正常工作，
   GPU 写 host 内存、GPU 读 host 内存都验证通过）。

**代价量化**（容器内微基准，`copy_to_uva` 等价操作）：

| 缓冲大小 | `UvaBuffer`（修复后） | `NonUvaBuffer`（修复前） |
| --- | ---: | ---: |
| 16 B | 1.6 µs | 48.4 µs |
| 8 KiB | 1.6 µs | 50.2 µs |
| 16 KiB | 3.7 µs | 59.4 µs |
| 1 MiB | 25.3 µs | 473.2 µs |

每次 decode step 有上百次这种调用 → 修复前约 15 ms/step 的同步开销（尽管部分与 GPU 重叠）。

**修复**：`-e VLLM_WSL2_ENABLE_PIN_MEMORY=1`（已在 `run_container.sh` 中）。

**效果**：

| 指标 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 单请求 decode | 83.10 tok/s | **112.27 tok/s** |
| decode 时 GPU 利用率 | 64~68% | 86~89% |
| decode 时容器 CPU | 96~108%（打满一核） | 56% |
| EngineCore 线程 CPU 占用 | 21.7 s / 25 s | 1.6 s / 25 s |

修复后瓶颈转为 GPU（87% 占用、175~188 W）；prefill 本来就比参考机快 25~40%。

## 本机硬件/链路特征（参考）

- GPU：`NVIDIA Graphics Device`，64 GiB，SM 上限 1695 MHz（decode 时稳定 1485 MHz），
  功耗上限 250 W（默认/上限 300 W），驱动 616.92，sm80。
- **PCIe：gen2 x8**（`pcie.link.gen.max=2`，当前 width 8）→ 实测 host↔device 带宽仅 **3.2 GB/s**
  （1 MiB H2D 310 µs、D2H 310 µs）；device↔device（HBM）约 580~700 GB/s。
- WSL2 下小内核 launch 开销正常：独立内核 4.7 µs/个，依赖链 5.9 µs/个，图中 50 个内核重放
  83 µs（1.7 µs/内核）→ 内核下发不是瓶颈。
- PLE SSD 侧：单行读 0.18 ms，16 行批量 0.44 ms，AIO 深度 256；PLE 路径每个 step 只占 ~2.3 ms（可被隐藏）。

## 环境要点（本机踩过的坑）

- **WSL2 的 nvidia runtime hook 会卡死**：`docker run --gpus ...` 会挂住。改用 CDI：
  `--device nvidia.com/gpu=all -e CUDA_VISIBLE_DEVICES=0`。
- **代理带宽约 10 MiB/s**，且 Docker Hub / HF 单连接极慢：
  - 模型用 `~/vllm/fdl.py` 分块并行下载（48 连接，`.fdl-state.json` 断点续传，
    `~/vllm/fdl_supervisor.sh` 自动重启，`~/vllm/resume.sh` 重启后一键恢复）。
  - 镜像用 `~/vllm/dl_blobs.py` 并行下 blob（token 5 分钟过期会自动刷新）再 `docker load`。
- Windows 侧若占用 GPU 显存，容器会在启动时因显存不足退出
  （`ValueError: Free memory on device cuda:0 ... less than desired GPU memory utilization`）。
  需要先腾空一整块 64 GiB GPU（可用 `nvidia-smi.exe` 查看 Windows 侧进程）。
- 校验脚本：`~/vllm/verify_ckpt.py`（头部/尺寸一致性）、`~/vllm/verify_sha256.py`（对照 Hub 的 sha256）。

---

# Prefill 变慢的根因排查（2026-09-28，全程实测；结论 21:5x 已更正）

## 0. 结论（更正版，先看这里）

**本机 prefill 没有坏，也没有配置问题。** 实测（同镜像、同默认配置 `QWEN_BATCH_TOKENS=2048`，
全程用仓库自带的 `benchmarks/bench_prefill.py` 的 prompt 生成方式，`seed=7391`，每次换 id 保证不走前缀缓存）：

| 场景 | 8192 token 冷 prompt | 32768 token 冷 prompt |
| --- | ---: | ---: |
| **稳态（Triton 内核已编译）← 正常值** | **2.17 s = 3779 tok/s** | **10.09 s = 3247 tok/s** |
| 参考文档 `docs/RESULTS.md`（09-19） | 3.11 s = 2636 | 15.37 s = 2132 |
| 前缀缓存命中（同前缀重复） | 0.95 s = 8587 | 0.99 s = 33033 |
| 容器刚起来、Triton 没编译（**假慢**） | 12.24 s = 669 | 52.07 s = 629 |

**两个真正的原因（都不需要改硬件、不需要买 SSD）：**

1. **Triton JIT 冷编译 —— 这就是“怎么突然变慢”的原因。** GDN（线性注意力）的 Triton 内核
   （`chunk_scaled_dot_kkt_fwd_kernel`、`recompute_w_u_fwd_kernel`、`l2norm_fwd_kernel2`、
   `_qsa_pre_indexer_kernel` 等，几十个变体 + autotune）是按 step 的**实际长度惰性编译**的，
   而且编译发生在**引擎 step 内部同步阻塞**。容器刚启动时，第一个长 prompt 要现场编译
   十几~几十个内核 → 同一个请求慢 **2~6 倍**；编译产物进 `/root/.triton/cache`，
   之后同形状就快了。
   证据：宿主挂载的 `~/vllm/triton_cache` 里编译时间戳与“慢请求”时刻一一对应
   （21:20:41 `_compute_local_logits_stats/_rejection/_resample_kernel`；21:29:56~21:30:10 二十个
   `chunk_scaled_dot_kkt_fwd_kernel` + `recompute_w_u_fwd_kernel` + 2 个 `*.autotune.json`；
   21:09:22 `_qsa_pre_indexer_kernel`）。同一容器同一配置编译完成后：8192 9.61 s→5.03 s、
   32768 36.43 s→20.2 s。
   `run_container.sh` 已把该缓存挂到宿主（`~/vllm/triton_cache`，108 MB）→ **重启不再重付**（关键修复）。
2. **`QWEN_BATCH_TOKENS=8192` 反而慢 2.0~2.35 倍**（我在 21:3x 提的建议是错的，作废）：
   cudagraph capture sizes 只到 2048，8192 的 chunk 超出所有已捕获尺寸 → 该 step 退回 **eager**。
   实测稳态：chunk 8192 → 8192 prompt 5.03 s (1609)、32768 prompt 20.1 s (1620)；
   chunk 2048 → 2.17 s (3779)、10.09 s (3247)。**保持镜像默认 `QWEN_BATCH_TOKENS=2048`。**

Decode 侧对照（确认没退化）：单流 ~110 tok/s、并发 4 聚合 ~260 tok/s、TTFT 0.07 s
（参考机 110.99 / 277.22 / 0.195）✔ 一致。

> ⚠️ 另一个关键现象：**跑过长 prompt（≥64K）之后，整个引擎会进入“慢状态”
> （prefill 慢 4.3~4.5×、decode 慢 1.7×），必须重启容器才恢复** —— 详见本文件第 7 节。

> 下面第 1~4 节是排查过程与原始数据，仍有价值（尤其“不是 PLE / 不是磁盘”的证伪）；
> **但第 5、6 节原来的结论（“8192 更快”“冷 prefill 只有 600~900 tok/s”）已被本节更正。**

## 1. 判别实验（同一容器，无需重启；`~/vllm/probe2.py`、`~/vllm/floor.py` 等价脚本）

| prompt | 冷（新 token ids，前缀未缓存） | **行缓存热**（前缀冷、PLE 行≈相同） | 前缀缓存全命中 |
| ---: | ---: | ---: | ---: |
| 512 | 0.80 s (640 tok/s) | — | 0.60 s (847) |
| 2048 | 2.42 s (845 tok/s) | — | 2.19 s (936) |
| 8192 | 11.87 s (690 tok/s) | **13.69 s (598 tok/s)** | 2.45 s (3344) |
| 32768 | 44.3~55.8 s (587~740 tok/s) | **53.8 s (609 tok/s)** | 4.66 s (7007) |
| 131072 | — | — | 15.3 s (8569) |

→ **行缓存热/冷几乎无差别（甚至略慢）**：PLE 行是否在内存里对 prefill 速度没有影响。
→ 前缀缓存命中时 ~7~8.6k tok/s：说明“引擎框架 + KV + 调度”这条路径本身很快（113 µs/token）。

## 2. PLE 车道本身的能力（容器内用生产库 `ple_ssd_io.so` + 生产 `PLESSDTable` 实测，`/tmp/lane2.py`）

- PLE 真实几何：**128 个分片 × (2500012, 160) bf16 = 320M 行 × 320 B = 95.37 GiB**，
  每 token `heads_per_ngram×(ngram_size-1)=8×2=16` 行。
- 一次 32768 行（=1 个 2048-token chunk）冷读：**318 ms = 103k 行/s**；
  128 次 ×256 行（prefetch 的真实形状）：**300 ms，不更慢**。
- 并发场景（256 行 read-ahead 与 32768 行 demand 抢同一把锁）：
  demand 每 chunk **262 ms**，16 chunk 共 4.2 s，**124.9k 行/s ≈ 7.8k tok/s**。
- 即：**PLE 车道能给 7.8k tok/s，引擎只用到 ~11k 行/s（~690 tok/s），差 11 倍。**
- 硬件层：4 KiB 随机 O_DIRECT 在 AIO depth≥256 下 ~121k IOPS（≈510 MiB/s）；prefill 期间
  `iostat /dev/sdd` r/s 平均 1.5k、%util 平均 1.5%（最高 72%）→ **盘基本闲着**。
- 放大：320 B 行按 4 KiB 页读 → 每行 ~4.4 KiB 设备流量，13× 读放大（这是 C 层 `ple_ssd_io.c`
  的对齐策略，不是盘的问题，也不是当前瓶颈）。

## 3. 运行时证据（py-spy，"cold 32k" 40 次全线程 dump）

- `MainThread` 100% 卡在
  `step_with_batch_queue → uniproc_executor.result → _finalize_prefetch(ple_ssd.py:480) → Future.result`；
  前缀缓存命中（无 PLE 计算）时则 100% 卡在 `get_output → torch.cuda.synchronize()`
  → 两次都是**等 GPU 把这一步算完**，不是等盘。
- `ple-ssd-prefetch_0` 线程：**97.5% 时间在 `_read_and_copy(ple_ssd.py:432)` 的
  `Event.synchronize()`（即等 ids 的 D2H 完成，`cudaEventSynchronize` 是自旋，所以它会打满一核）**，
  只有 2.5% 真在 `read()`。→ 读盘不是关键路径，ids 的 D2H 事件在等主流的队列排空。
- `ple-prompt_0`（read-ahead）：100% 空闲（已提前读完）。
- GPU：util 100% 但只有 **~100 W / 250 W**，SM 1485/1695 MHz，**无任何 throttle**（37 °C）；
  实测算力 194 TFLOPS bf16、HBM 1.55 TB/s。→ 有大量余量，属于**延迟/串行受限**。
- 单 chunk 成本随上下文缓慢增长（2.42 s@ctx2k → 3.49 s@ctx24k），说明 attention 只占 ~10%，
  其余 ~90% 是**与上下文无关的 per-token 权重/MoE 路径**（~1.1 ms/token，约峰值的 2%）。

## 4. 已否定的假设

| 假设 | 结论 |
| --- | --- |
| PLE 分片碎片化（3182 extents）/ DRAM-less SSD IOPS 不够 | 否。行缓存热也一样慢；车道 121k IOPS；盘 %util 1.5% |
| 该上企业盘 / Optane / 重新顺序写 95 GiB 分片 | 否，别花这个钱和时间 |
| `wsl --mount` 裸分区绕过 VHDX | 无需（不是 I/O 瓶颈） |
| 09-27 的 3769 tok/s 是真·冷 prefill | 否。那是前缀/行缓存命中（本次实测 7007 tok/s @32k 命中） |
| read-ahead 窗口太大抢锁 | 有影响但不是主因（256 行批次实测不慢） |

## 5. 下一步 A/B（每条约 7 分钟重启，改 `~/vllm/run_container.sh` 的环境变量）

1. ~~`QWEN_BATCH_TOKENS=8192`~~ **已实测否定**（第 0 节第 2 条：chunk 8192 稳态慢 2.0~2.35×，
   因为 8192 > cudagraph capture sizes(2048) → 退回 eager）。**保持 2048。**
   （真要试大 chunk，必须同时放大 `QWEN_CAPTURE_SIZES`；那会吃掉 KV cache（只剩 3.77 GiB
   < 262144 上下文所需 7.25 GiB）→ 引擎拒启，除非把 `QWEN_CONTEXT` 降到 ~131072。
   而 2048 档已经比参考文档更快，不值得用上下文换。）
2. **冷启动后的 warm-up（推荐，不用重启）**：容器起来后先发几个“废请求”，长度覆盖你实际会用到的
   档位（如 2k / 8k / 32k / 最大档位），各 1 次，让 Triton 编译 + autotune 走完，再开始真正的工作。
   这样只有第一个请求慢，之后稳定 3.2~3.8k tok/s。
3. `QWEN_SSD_PREFETCH=0` / `QWEN_SSD_CACHE_MB=4096`：优先级很低。第 2、3 节已证明 PLE 车道能给
   7.8k tok/s，而稳态 prefill(3.8k tok/s) 只用 ~61k 行/s，盘 %util ~1.5%。
4. 若 1 有效但还不够：拿本文件第 2、3 节的实测数字找 patch 作者对账（参考机同 GPU 同镜像
   `docs/RESULTS.md` 报 2132~2636 tok/s @32k，本机 587~740）。
5. **给作者的具体建议**（PLE 路径在 WSL2/长延迟下的每 chunk 同步）：
   `start_prefetch` 里的 `_copy_ready.synchronize()`（每个 chunk 把 CPU 排空一次）+ 
   `wait_event(_previous_use)` + `wait_stream(current_stream)` 形成“上一 chunk 算完 → 才 D2H ids →
   才读盘 → 才能算下一 chunk”的串行链；建议把 ids 的 D2H 提到 chunk 开头、
   用 `record_stream`/独立流取代 `wait_stream`，并去掉主线程上的 `_copy_ready.synchronize()`。

## 6. 对“长文档 compaction”工作流的实操建议

- **用镜像默认 `QWEN_BATCH_TOKENS=2048`**（第 0 节），并保持 `~/vllm/triton_cache` 持久挂载。
- **重启容器后先 warm-up**：先发 1 个 8k、1 个 32k、1 个你实际最大档位（如 160k）的废请求，
  再开始正式 compaction。否则第一个长 prompt 会慢 2~6 倍（Triton 冷编译），很容易误判成“机器变慢了”。
- **保持前缀稳定**（同文档同前缀）：prefix cache 命中时 32k ≈ 1 s（33k tok/s），这是最大杠杆；
  前缀一变 → 全额重算 → 稳态 ~10 s / 32k（3247 tok/s）。
- 预期稳态：8k ≈ 2.2 s、32k ≈ 10 s，prefill 速率 3.2~3.8k tok/s（比参考文档 2132~2636 还快）。
- 与磁盘、碎片、缓存调参无关（第 2、4 节）。

---

# 7. 长 prompt 悬崖 + “慢状态”（2026-09-28 22:0x~22:5x 实测）

> **2026-09-29 更正（重要）**：7.1 的“32K→64K 悬崖”是**测量假象**——那次 64K（138 s）是接在两次 160K
> 之后测的，引擎已经在“慢状态”里了。干净容器里 **64K 真实文本 = 19.6 s（3484 tok/s），完全线性**。
> 真正可复现的是 7.2 的“慢状态”（长 prompt 之后整个引擎变慢，重启才恢复）；更正后的数字见 7.5。

## 7.1 稳态下的 prompt 长度曲线（chunk 2048，全新 token ids，前缀不命中）

| prompt | 耗时 | tok/s |
| ---: | ---: | ---: |
| 8192 | 2.22 s | **3695** |
| 32768 | 9.58 s | **3420** |
| 65536 | 19.6 s（真实文本，干净容器） | **3484（无悬崖）** |
| 100K（CJK 文本） | 51.8 s（干净容器第一次请求） | 1928 |
| 186K（ASCII 代码） | 563.8 s（干净容器第一次请求） | 331 |
| random 64K（慢状态下测） | 138 s | 473 |
| random 131K（慢状态下测） | 298 s | 440 |
| random 160K（慢状态下测） | 427~500 s | 328~383 |

→ 64K 以内基本线性（~3400~3500 tok/s，真实文本）。100K 以上明显退化（1928 tok/s @100K、331 tok/s @186K），
  但注意后两行是“慢状态”下的数字，不是长度的固有成本（见 7.2）。

## 7.2 跑过长 prompt 后，整个引擎进入“慢状态”（重启才恢复）

同一容器、同一配置：

| 项目 | 快（刚重启，只跑过 ≤32K） | 慢（跑过 64K/131K/160K 之后） |
| --- | ---: | ---: |
| 全新 8192 prefill | 2.22 s / 3695 tok/s | 9.89 s / 829 tok/s（**4.5×**） |
| 全新 32768 prefill | 9.58 s / 3420 tok/s | 41.5 s / 789 tok/s（**4.3×**） |
| decode 单流 | 59.8~66.1 chunk/s（≈110 tok/s） | 34.5~40.8 chunk/s（≈62~73）（**1.7×**） |
| 前缀命中 32K | 0.95 s | 3.63 s（**3.8×**） |

- **不是 PLE 车道退化**：同一时刻在容器内用生产库单独实测，车道仍是 **100.6k 行/s**
  （一次 32768 行 = 325 ms），与快状态（103k 行/s）一样。
- 但设备侧明显更忙：`iostat /dev/sdd` 随机读 **r/s 1.5k → 52k（30×）**、`r_await` 0.26 → 1.1~1.6 ms、
  %util 最高 39%（正常情况下 prefill 时 %util ~1.5%）。
- py-spy：主线程每个 chunk 卡在 `_finalize_prefetch(ple_ssd.py:480)` 等 prefetch future；
  `ple-ssd-prefetch_0` 自旋在 `Event.synchronize()`（`cudaEventSynchronize`，持掉一个核，EngineCore CPU 93.9%）。
- GPU：util 90~100% 但功率只 **86~152 W**（饱和应 249 W）→ 延迟/串行受限，不是算力受限。
- **重启容器立即恢复**（实测：重启后第一个请求 8192 = 2.22 s、32768 = 9.58 s）。

**目前最可能的机制（待 A/B 确认）**：512 MiB 行缓存（1.6M 行 ≈ 100K token）被长 prompt 的行填满且一直抖动
（160K token 需 2.6M 行），demand 读与 16384-token read-ahead 互抢一锁 + 互抢盘，
把每 chunk 的 PLE 握手（`_finalize_prefetch`）从 ~0.1 s 抬到 ~2 s，连带把 decode 也拖慢。

## 7.3 下一步 A/B（各 ~7 分钟重启，只改一个变量）

1. **`QWEN_SSD_CACHE_MB=4096`**（4 GiB 行缓存 = 8M 行 ≈ 500K token，160K prompt 的行能全放下）：
   预测“慢状态”消失或大幅减轻。
2. **`QWEN_SSD_PREFETCH=0`**：关掉 read-ahead，让单车道专注 demand 读。
3. 若 1/2 有效，把这组数字 + 第 2、3 节证据给 patch 作者（PLE 每 chunk 同步链 +
   行缓存容量与 read-ahead 的相互作用）。

## 7.4 给 compaction 工作流的铁律

- 一次全新 160K → **7~8.5 分钟**；同前缀重复 → **秒级**（prefix cache，33k tok/s）。
- **长 prompt 之后接短交互/低延迟任务前，先重启容器**（~6 分钟）。
- 重启后第一枪就是全速（Triton 缓存已在宿主，实测 8192=2.22 s / 32768=9.58 s，不需额外 warm-up）。

## 7.5 更正与最终数字（2026-09-29 复测）

**可复现的（必现）**：跑过一次 ~100K+ 的全新 prompt 之后，**整个引擎变慢，重启才恢复**
（3 次独立复现：随机 id 1 次、真实文本 2 次）：

| 项目 | 干净容器 | 被“污染”后 |
| --- | ---: | ---: |
| 全新 8K | 2.25 s / 3643 tok/s | 9.29~13.05 s / 628~882 tok/s |
| 全新 32K | 9.88 s / 3318 tok/s | 41.5~54.9 s / 572~789 tok/s |
| decode 单流 | ≈110 tok/s (62~67 chunk/s) | ≈62~73 tok/s (34~41 chunk/s) |
| 每个 token 的额外成本 | — | **+0.5~0.9 ms/token**（1~2 chunk 的短 prompt 不受影响） |

排除项：**行缓存 512 MiB→4096 MiB 无效**（117K 仍是 870 tok/s）；无内存压力
（MemAvailable 53 GB、SwapFree 100%、VmSwap 0）；PLE 车道同时刻单独实测仍 100.6k 行/s。
`iostat`：被污染时随机读 r/s 1.5k→52k、r_await 0.26→1.1~1.6 ms；GPU util 90~100% 但功率只 86~152 W。

**尚未解释**：干净容器**第一次**长 prompt 也偏慢（100K CJK=1928 tok/s；186K ASCII=331 tok/s），
比用户 09-27 记录的“200K ≈ 1.3~1.5k tok/s + decode 110”差 1~4 倍。差异可能在**测量方法**或
**内容**（我的语料是 Python 代码/中文文档，与真实长文档的 ngram 重复率不同 → PLE 行缓存命中率不同）。
要对齐需要：用户昨天那条 200K 的原文 + 测法（端点/是否流式/一次还是多轮）。

**实操铁律**：长 prompt 放会话最前面跑；跑完长 prompt 要低延迟就先重启容器（~6 分钟，重启后第一枪
就全速：实测 8K=2.25 s / 32K=9.88 s）。保持 `QWEN_BATCH_TOKENS=2048`（默认）。

# 8. "慢状态"根因定位：PLE/SSD 无罪，代价在 GPU 侧 prefill（2026-09-29 00:2x~00:3x 实测）

## 8.1 单变量 A/B：关掉 PLE 读预取**不能**解决慢状态（Astra 建议的 #1 假设被否）
`QWEN_SSD_PREFETCH=0 PLE_PATCH=~/vllm/ple_ssd_instr.py ./run_container.sh`（容器内 ple_ssd.py 挂载为带计时版本）：
| 请求 | 时间 | 速率 |
|---|---:|---:|
| 干净 8K | 2.97 s | 2756 tok/s |
| 干净 32K | 11.25 s | 2912 tok/s |
| 长 prompt（真实英文 142983 tok） | 270.97 s | 528 tok/s |
| 之后 fresh 8K | 10.28 s | 797 tok/s ← **仍然中毒** |
| 之后 fresh 32K | 44.73 s | 733 tok/s ← **仍然中毒** |

→ 预取不是触发源。**但预取对干净状态有益**（干净 8K：预取开 2.25~2.31 s / 关 2.97 s，约 +24%），所以生产配置应保持默认（预取开）。

## 8.2 每 chunk（2048 token / 32768 PLE 行）分段计时（引擎内部插桩，单位 ms）
| 阶段 | 干净 32K | 中毒 32K |
|---|---:|---:|
| chunk 墙钟 | **546** | **2305** |
| `io`（native AIO 真实落盘读） | 191 | 209 ← **设备满速未退化** |
| `W_ids_sync`（worker 等"GPU 完成 id 交接前的工作"） | 257 | **1826** ← 主因 |
| `W_table_read`（读+evict+H2D 入队，worker 侧） | 221 | 267 |
| `M_wait_pending`（引擎线程阻塞等 PLE 结果） | 477 | **2092（占 91%）** |
| `lookup`/`fill`/`join`（Python） | 6/9/10 | 15/24/13 |
| `cache_len` | 1198372（满） | 满 |
→ 落盘读 ~157k 行/s（32768 行 191~209 ms），Python 开销 ~50 ms/chunk。**PLE/SSD 全程满速，锅不在它。**
→ 干净状态下 PLE 读占整个 chunk 的 40%（221/546）且几乎与 GPU 串行（depth-1 握手：读期间 GPU 无事可做）。

## 8.3 代价在 GPU 侧 prefill，且 decode 不受影响
- py-spy 全线程采样（中毒 8K 期间，14 s）：MainThread（= 执行线程）
  **73% 卡在 `Future.result() ← _finalize_prefetch ← graph.replay()`**（即 PLE 断点处，正在回放 CUDA 图，**不是 eager**），18% 空闲等请求，1.4% 在 GDN 内核；
  `ple-ssd-prefetch_0`：58.4% 等 `_ids_ready`，10.4% 真读盘。
- **并发 GEMM 探针**（另一进程跑 8192³ bf16 GEMM，5.3 ms/次，正常 206~211 TFLOPS）：中毒期间 GEMM 每 ~3.4 s 就有一次 **~1.9~2.7 s 完全拿不到 GPU**，其余时间满速 206~211 TFLOPS → 引擎的每次 prefill chunk 独占 GPU 约 2 s。
- 同一探针里 `max_tokens=64` 的 decode 为 **0.98 s / 64 token = 15.3 ms/token（干净水平）** → **中毒只影响 prefill**。
- 单独测 GPU：bf16 8192³ = 194~207 TFLOPS、D2D 1.37 TB/s（中毒时也是满速）→ **GPU 硬件从未变慢**。

## 8.4 本轮一并排除（都是实测，不是猜测）
- GPU 时钟/功率：干净/长/中毒三段全程 **1485 MHz**；长 prompt 期间才出现 0x4 SW power cap（峰 268 W）。
- Triton JIT/autotune 反复编译：`~/vllm/triton_cache` 在中毒期间**新增文件数 = 0**；引擎日志只有 3 条一次性 JIT 警告（spec-decode 采样内核，16:37:18）。
- 重复计算/抢占：`request_prefill_kv_computed_tokens_sum == prompt_tokens_total == 224903`（无重算）。
- KV/prefix 残留：长跑后 `kv_cache_usage_perc=0`、`prefix_cache_hits=0`。
- 主机内存/swap（RSS 4.43 GB、VmSwap 0）、空闲 CPU（0.5%，无自旋线程）、线程数 55（无泄漏）、AIO context 退化（中毒时独立 reader 仍 140k 行/s）、prompt 内容（随机 id / 英文 / 中文同速）、chunk 大小（两态都是 2048）。

## 8.5 结论与下一步
**结论**：慢状态 = 每次 *prefill* chunk 里被回放的 GPU 工作从 ~0.26 s 变成 ~1.8~2.4 s（7~9×，形状/内核/配置都没变、没有重编译、没有重算、时钟没降、GPU 单测满速），一次 64K~100K+ token 的请求把它"翻"过去且只有重启能翻回来。PLE 断点的 depth-1 握手把这段 GPU 时间**直接暴露**在引擎线程上（91% chunk 时间阻塞），所以看起来像"PLE 慢"。剩下的候选都在**回放的图内部**：MoE prefill（humming indexed GEMM）路径、GDN chunked-prefill Triton 内核、12 层 full-attention prefill、以及 KV/mamba `align` 的元数据内核。decode 不中招说明它与 prefill 专属内核/元数据有关。
**下一步（唯一能"点名"的工具）**：镜像里 **有 `nsys` 和 CUPTI**（`/usr/local/bin/nsys`、`nvidia/cu13/lib/libcupti.so.13`）→ 用 nsys 包住 entrypoint 重启一次，trace「干净 8K → 100K 长 prompt → 之后 8K/32K」三段，`nsys stats` 出按内核的耗时对比，即可指认是哪个内核/哪类内核变慢。诊断用的 `PLE_PATCH=` 挂载已加入 run_container.sh；插桩文件 `~/vllm/ple_ssd_instr.py`（含 `PLESTAT` 每 2 s 一行日志）。

# 9. 慢状态定位到注意力路径（2026-09-28 17:19~17:33 实测，一次重启 + 无 CUPTI 的图内 event 计时）

## 9.0 方法（本环境 CUPTI 全废：torch.profiler → CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED(42)；nsys → importer error 无 .nsys-rep；ncu → ERR_NVGPUCTRPERM）
改用 CUDA event：**capture 期**在模块边界记录 `torch.cuda.Event(enable_timing=True, external=True)`，
这些 record 节点被烘进图、**回放时执行**（先用独立小实验验证：1 次 2048³ mm = 1.665 ms、20 次 = 33.14 ms，且每次 replay 数值都会更新），
回放后延迟一步读 `elapsed_time`。另在 `BreakableCUDAGraphCapture.replay()` 里对每个 segment 用普通 event 对 + 主机时钟计时。
实现：`~/vllm/cg_instr.py`（挂到容器内 `vllm/compilation/cg_instr.py`，由插桩版 ple_ssd.py 在 import 时 install()），
`run_container.sh` 新增 `CG_PATCH=` + `CG_INSTR=1`；`~/vllm/ple_ssd_instr.py` 继续提供 per-chunk PLE 计时（PLESTAT）。
注意：`Qwen4ExpDecoderLayer` 等模块的 forward 若跨 graph break，其 event 区间会包含断点间的间隙（伪影）；**段级计时不会**，
所以段级数据为主、模块级为辅。

## 9.1 图结构（实测）
- prefill 用 piecewise 图：**103 个 segment**（≈每层一段 + eager 断点），每个 capture 记录 **144 个模块区间**（48 层 × {decoder, attn, moe}）。
- decode / MTP draft 用 FULL 图：**3 个 segment**，同样 144 个区间。
- 启动共 ~120 次 capture，每个 capture 144 对 event（池 24576 对，25.8 ms 建好）。

## 9.2 干净 vs 中毒：主模型每个 prefill chunk（2048 token）的 GPU 时间（毫秒）
| 测试轮 | 总量 | ATNQ（12 层 QSA 全注意力） | GDN（36 层线性注意力） | MOE（48 层） |
|---|---:|---:|---:|---:|
| 干净 | **389 / 475** | 66 | 217 | 112 |
| 96K 触发中 | 6674 | 2483 | 912 | 140 |
| 中毒后 | **2518 / 2617** | 1085 | 1300 | 190 |
→ 每个 chunk 从 ~0.43 s 涨到 ~2.6 s（**6×**），增量集中在两种注意力上；MoE（纯权重 GEMM）几乎不变（+1.2~1.7×）。

## 9.3 更关键的证据：**decode 每步也慢 8.9×，且增量 100% 在 QSA 注意力**
| 每步 | 干净 decode | 中毒 decode |
|---|---:|---:|
| 步总 GPU 时间（FULL 图 3 段） | **12.4 ms** | **110.5 ms**（+98 ms/步） |
| ATNQ（12 层） | 5.6 ms | **80.6 ms**（+6.2 ms/层/步！） |
| MOE（48 层） | 3.6 ms | 4.8 ms |
（MTP draft 单层模型同样：21.5 → 107.4 ms，ATNQ 13.4 → 77.2）
→ 一个只有 1~2 个 token 的 decode 步，QSA 注意力每层要 6 ms —— 这不是"注意力算得多"，而是**在扫描/等待什么巨大的东西**（100% SM util 但只有 ~100 W = 访存/自旋型）。

## 9.4 端到端 TTFT（fresh 随机 id，同一容器同一次运行）
| prompt tokens | 干净 | 中毒后 |
|---|---:|---:|
| 512 | 0.447 s (1145 tok/s) | 0.964 s (531) |
| 2048 | 0.662 s (3093) | 2.364 s (866) |
| 8192 | 2.006 s (4083) | 11.947 s (686) |
| 32768 | 9.828 s (3334) | 52.697 s (622) |
96K 触发请求本身：49.6~59.8 s（1981 tok/s，干净 32K 是 3334）。
→ **中毒不是"长 prompt 慢"，而是"所有后续步都慢"**（decode 也慢），之前"prefill-only"的结论被本次干净测量推翻。

## 9.5 嫌疑路径（full attention = 自定义 QSA 稀疏注意力）
`models/qwen4_exp/nvidia/qsa.py`：`Qwen4ExpQSAAttention` + `Qwen4ExpQSAFlashAttentionImpl`（backend `QWEN4_EXP_QSA_TRITON`，`is_sparse()=True`），
`nvidia/indexer_qsa.py`：`QSAIndexer`（`token_topk = config.indexer_budget`、`compress_ratio`、`visible_blocks`、`skip_topk`（MTP 复用 step0 的行）），
`nvidia/ops/qsa_pre_indexer.py`：`_qsa_pre_indexer_kernel`（融合 RMSNorm+RoPE+压缩 K，**原子累加进 circular state/compressed cache**），
`common/qsa_cache.py`：压缩 KV / state cache 与 `visible_blocks` 计算（`qsa_cache.py:212-275, 390-440`）。
之前 Triton 首次编译日志里出现过 `_qsa_pre_indexer_kernel`、`_compute_local_logits_stats_kernel`、`_rejection_kernel`、`_resample_kernel`
（top-k **采样**：local logits stats + 拒绝采样 + 重采样）——拒绝采样式 top-k 的成本对数据敏感，且可能退化成自旋。

## 9.6 已排除（本轮实测）
PLE/SSD（读盘满速、预取开关无关）、GPU 时钟/功率、GPU 算力（194-207 TFLOPS 未变）、Triton 重编译、重复计算/抢占、
KV/prefix 指标残留（usage 0）、主机内存/线程、chunk 大小、prompt 内容、CUDA 图机制本身（同样的图在回放，段级 GPU 时间就是变慢了）。

## 9.7 下一步（一次重启）
把 event 插桩下沉到 QSA 路径内部（挂载改过的 `indexer_qsa.py`/`ops/qsa_pre_indexer.py`/`qsa.py`）：
在 (a) q/k 投影+norm+rope、(b) state/compressed cache 原子更新、(c) local-logits-stats+rejection+resample 采样、(d) top-k 选择、
(e) 注意力主体、(f) metadata 构建 这几组前后各埋一对 baked event，就能指出是哪一组在中毒态暴涨（以及是否随 context 增长）。
同时验证一个可操作假设：`skip_topk`/`indexer_budget`/`compress_ratio` 相关配置（`config.json` 里）能否作为绕过手段。

## 9.8 关键判别（2026-09-28 17:45~17:47，无需重启）：中毒成本 ∝ **候选/元素个数** 的 **~13× 单价**，不是"固定扫描大缓冲区"
Astra 的纠正 + 一次极便宜的实测（先 96K 触发，再跑 256-token 小上下文请求解码 16 token）：
| 解码每步（FULL 3 段图） | 干净 | 中毒 |
|---|---:|---:|
| 256-token 上下文（可见压缩候选≈64 < k=512，top-k 走平凡分支） | 12.7 ms（ATNQ 5.8） | **15.1 ms（ATNQ 11.1，仅 +5.3）** |
| 2K~32K 上下文（可见候选 512~8192，走真正选择路径） | 12.7 ms（ATNQ 5.8） | **110.5 ms（ATNQ 80.6，+75）** |
| prefill chunk 512 token | ~388-475 ms(2048tok) | 702 ms（ATNQ 134 + GDN 484） |
| prefill chunk 2048 token | 389/475 | 2518/2617 |
→ 中毒成本随**候选数/元素数**线性放大（每元素约 13×），**不是**固定扫一个大缓冲区；
→ 我原先的 H1"按历史最长上下文固定扫描宽度"被**否掉**；"可见长度被污染成 96K 规模"也被否掉（否则 256-token 请求也要 ~80 ms/步）。
→ 结论方向：QSA（+GDN）路径上**每个候选/元素的访存代价**涨了 ~13×（访存/延迟型），而 MoE（流式读权重）不受影响 → 指向"缓存/状态数据的访存路径"而非算力。

### Astra 对源码的三处纠正（已接受，避免白跑重启）
1. `common/qsa_cache.py` 的 `REQUEST_SCAN_SIZE = next_power_of_2(request_capacity)` 里的 capacity 来自 `max_num_seqs`（=4），是**请求维度**的扫描宽度，不是上下文维度 → 不支持 H1。
2. `_compute_local_logits_stats_kernel`/`_rejection_kernel`/`_resample_kernel` 属于**投机采样**（`v1/worker/gpu/spec_decode/rejection_sampler_utils.py`），**不是 QSA 的 top-k**；QSA 在 sm_80 上调用 `torch.ops._C.persistent_topk`，选 **512 个压缩位置**再展开成 ≤2051 个 token 位置。
3. 融合 pre-indexer 现在是 reduce/store（不是原子累加）；且 `persistent_topk` 只接受 **k=512/1024/2048** → 我原计划的 `indexer_budget=256` / `compress_ratio=8` 覆盖是**非法的**（会直接失败），已放弃。
另：`persistent_topk.cuh` 有 inter-CTA 自旋等待，其协作式 radix 路径在**压缩候选 > 32768（≈131072 逻辑 token）** 才启用 —— 而 96K 触发只有 24576 候选，**低于**该阈值，所以"radix 阈值"也不是触发点。

## 9.9 QSA 路径观测轮（2026-09-28 18:56~19:00）
本轮保持生产配置：chunk=2048、prefetch=16384、gpu_mem_util=0.96；新增 QSA 文件挂载只用于观测，实验结束后已卸载并恢复默认容器。

### 主实验：干净 → 96K → 中毒
| fresh prompt | 干净 | 中毒 |
|---:|---:|---:|
| 512 | 0.433 s / 1182 tok/s | 1.038 s / 493 tok/s |
| 2048 | 0.655 s / 3126 | 2.544 s / 805 |
| 8192 | 2.012 s / 4072 | 12.720 s / 644 |
| 32768 | 9.838 s / 3331 | 56.498 s / 580 |
| 98304 trigger | 60.344 s / 1629 | — |

图内事件复现：干净 2048-token chunk 约 389~475 ms；96K 触发期间约 6742 ms；中毒后约 2696~2755 ms。解码图中 ATNQ 约从 5~6 ms/step 升至 68~90 ms/step，MOE 约 5 ms，和此前结果一致。

### QSA 子阶段插桩结果
`ops/qsa_indexer.py` / `ops/qsa_pre_indexer.py` / `ops/qsa.py` 已成功挂载，容器初始化和主实验均成功。QSA 子阶段事件没有进入 `CGSTEP` 汇总：这些调用落在 QSA 的 eager-break 路径，当前 `cg_instr.stage()` 只在 `_CAPT_ON` 的图捕获区接受事件；因此本轮没有把 `PAGED_LOGITS/TOPK/PRE_INDEXER/ATTENTION` 拆开，不能据此判断某一个 QSA kernel。

该插桩缺口不改变主结论：现有可复用的模块级 CUDA 事件已经在同一轮确认 ATNQ 是主要增量，GDN 是次要增量，MOE 基本不变。后续若继续拆 QSA，应改为 eager-segment 内的独立事件生命周期，而不是再次盲目重启。

### 收尾验证
实验完成后容器已重启为默认配置，容器 `3cf5a996ba49`，`/health=200`；新鲜 8192-token prefill 为 `2.406 s / 3405 tok/s`。当前服务是干净态。

# 9.10 图内事件第二轮 + 时钟/功率定性（2026-09-29 08:3x~09:5x）

本轮修好了两处观测缺口，并第一次拿到**请求窗口内 10 Hz 的时钟/功率**证据。结论一句话：
**中毒不是“活变多了”，也不是降频/降算力，而是同一份活在 SM 上“停顿”着做完（满时钟 + 低功率 + 慢）**，且停顿只出现在**模型自己的 prefill 图段**里。

## 9.10.1 修好的观测缺口
- `cg_instr.stage()` 在 eager-break（replay 期）的事件生命周期 + `_cg_stagepairs` 合并进 `pairs` → QSA 子阶段事件第一次真正进入 `CGSTEP`。
- QSA 各阶段加了 `qa()` 启动参数指纹（受 `sync_ok()` 保护，捕获期绝不读回）→ `QSALOG` 行。

## 9.10.2 纠正：§9.3/§9.9 的“decode 也慢 8.9×”是误读
那批 `nseg=3, nl=1, ATNQ≈68~90 ms` 的 CGSTEP 行是 **MTP draft 的 prefill**，不是 decode。
本会话实测真 decode（28-token 提示，生成 64 token）：**21.1 ms/token（中位）/21.2（均值）**，与干净态 19.9~21.3 一致 → **中毒是 prefill 专属**。

## 9.10.3 主实验（本会话，干净容器 vs 同一容器中毒后）
| fresh prompt（全新随机 ids） | 干净 | 中毒 |
|---:|---:|---:|
| 512 | ~0.22 s | 0.714 s |
| 2048 | 0.646~0.681 s（3172 tok/s） | 1.969~2.128 s（1040 tok/s） |
| 8192 | ~2.39 s | 10.068 s（814 tok/s） |
| 32768 | ~10 s | 44.152 s（742 tok/s） |
| 98304（触发本身） | 49.969 s（1967 tok/s） | 第二次跑 56~64 s → **触发请求自己不变慢** |
| decode | ~19.9~21.3 ms/token | 21.1 ms/token（不变） |

## 9.10.4 决定性事实 1：同一请求的 QSA 启动参数**逐字节相同**
同一个 fresh 2048-token 请求，干净 vs 中毒：

| 指纹 | 干净 | 中毒 |
|---|---|---|
| ATTN | `nq=2048 sel_w=2051 n_tiles=65 n_splits=1 warps=1 block_n=32 cnt_max=2048 cnt_sum=2.098e6 ti_max=2047` | 完全相同 |
| PLOG_PREFILL | `nq=2048 w=512 g1=32 g2=1 vis_max=512 vis_sum=5.238e5` | 完全相同 |
| TOPK | `w=512 vis_sum=5.238e5` | 完全相同 |
| METADATA | `tok=2048 pre=1 seq_max=2048 vis_max=1280 vis_sum=1.311e6` | 完全相同 |
| 物理 KV block | `bt0_max=9.33[2..13] bt0_min=3.33 bt0_jumps=31.3` | `bt0_max=42.5~50[17..83] bt0_min=1~8.5 bt0_jumps=1.5~2` |

触发请求（96K）的 `w≈2.46e4 / vis_sum≈4.09e7 / ti_max≈9.8e4 / cnt_max=2050` 属于**长上下文 chunk**，不属于中毒后的 fresh 请求。
⇒ 正式排除：**“活变多了”、“元数据陈旧/错位”、“可见长度被锁死在 96K”、“radix 阈值”**。唯一的物理差别是**块的物理位置**（中毒后更连续、但落在池内更靠后的位置）。

## 9.10.5 决定性事实 2：代价分布（同形 2048-token prefill，图内事件，ms）
| 类目 | 干净 | 中毒 | 倍率 |
|---|---:|---:|---:|
| sum_gpu | 679.6 | 2613.0 | 3.8× |
| **QSA ATTENTION（每层）** | **2.2** | **30.0** | **13.6×（12 层全部 30.0，极其一致）** |
| ATNQ 合计 | 85.8 | 784.0 | 9.1× |
| GDN 合计 | 422.0 | 1588.5 | 3.8× |
| MoE | 134.5 | 197.2 | 1.5× |
| PAGED_LOGITS/TOPK/PRE_INDEXER | ~0.0~0.5 | ~0.0~0.5 | 1× |
| sum_host（replay 循环内宿主时间） | 647.7 | 2488.0 | 3.8× |
| MTP draft prefill（nl=1/nseg=3） | 13.1 | 94.2 | 7.2×（其单个注意力 2.2→30.0） |

103 个图段**均匀**膨胀 ~7×；但 `GDN_FUSED`（融合核本身）**不变**，变的是整个 GDN 模块。
⇒ 只有**「按索引/状态取数」**的两条路径被打（QSA 稀疏注意力 + GDN 递推），**流式权重路径（MoE）几乎不动**。

## 9.10.6 决定性事实 3：满时钟 + 低功率 + 无节流（请求窗口内 10 Hz 采样）
| 窗口 | SM 时钟 | 功率 | 节流标志 |
|---|---|---|---|
| 96K 触发（干净态） | 忙时均值 1461 MHz | 忙时均值 136 W / 峰值 301 W | `0x4 SW Power Cap` 命中 97/602 采样 |
| 中毒 fresh 2048 | **均值 1477 MHz（峰值 1485）** | **均值 105 W / 峰值 158.8 W** | **无任何节流标志** |

⇒ **时钟假设彻底死亡**（这次是请求窗口内的定频采样，不是相位采样）。满时钟 + 只花 105 W + 干得慢 3× = **SM 在停顿**（等内存/等依赖），不是降频、不是算力下降。

## 9.10.7 决定性事实 4：PLE/SSD 的等待完全由“上一段 segment 的 GPU 时间”解释
PLESTAT：设备侧 AIO 时间 `io` 干净 191 ms / 中毒 209 ms（不变）；每层 `W_ids_sync` ≈ `M_wait_pending` ≈ 上一段 segment 的 GPU 时间（干净 5.4 ms/层 vs segment 5.9；中毒 38 ms/层 vs segment 30~42）。
链条是 `[segment N-1] → ids D2H → SSD 读 → H2D → [segment N]`，PLE 只是**被动等模型**。
（`ple_ssd_prefetch_tokens=0` 的 A/B 见 §8.1：关预取**不能**修中毒，且干净态慢 24%。）
注：PLE 仍是**总量**的大头——干净态每个 2048-token chunk 的设备读时间约 190 ms，占 400 ms chunk 的近一半；但它不是“中毒”的来源。

## 9.10.8 决定性事实 5：独立探针在中毒态完全正常
独立 CUDA 上下文（~300 MiB 足迹，两种状态都跑）：bf16 8192³ 156~200 TFLOPS；streaming copy 1541~1568 GB/s；4/16/64 KiB 页散射 316~329 GB/s；随机元素 gather 47~58 GB/s —— **干净/中毒一致**。
⇒ 裸内存/算力通路健康；中毒只发生在**引擎自己的**内核里。

## 9.10.9 被工程现实挡掉的 A/B：VRAM 余量
`QWEN_GPU_MEMORY=0.88` **起不来**：`Available KV cache memory: 4.1 GiB` < 需要的 `7.25 GiB`（max_model_len 262144）。
由此测出真实账本：权重 47.32 GiB + 图池 2.23 GiB + 其他 ~2.7 GiB + **KV 池 ~9.2 GiB** = 61.4/64 GiB —— **日志里的 “kv cache memory in use 9.22 GiB” 是 KV 池总量，不是 96K 请求的占用**（之前误读）。
可用窗口只有 ~0.93~0.96，做不了“多留余量”的单变量实验。

## 9.10.10 本轮能确认的结论
1. **触发**：一次 ≥64K~96K token 的 prompt（分 chunk 处理）。
2. **效果**：之后**每个** prefill chunk 慢 ~3.1×（2048：0.65→1.97 s），decode 不变（21.1 ms/token），**触发请求自己不变慢**。
3. **位置**：模型自己的**每层图段**里等量工作耗时 ~7×；增量集中在 **QSA 稀疏注意力（13.6×）** 与 **GDN 递推（3.8×）**；MoE 1.5×；PLE/SSD 设备时间、宿主 launch 路径、独立探针都正常。
4. **机制层面**：**SM 停顿**（满时钟 1477 MHz、无节流、105 W、util 57%），不是降频/算力/带宽/宿主/SSD/“活变多”。
5. **最一致的剩余解释**（**未证明到驱动层**）：长 prompt 的分配/工作集冲击（0.96 的 VRAM 占用、WSL2 上与 Windows 显示共享 64 GiB 设备）之后，引擎**自己的长生命周期缓存（KV/QSA/GDN state）在设备侧的页映射/局部性变差**，使“带索引的 gather”型内核每次访存多付 ~20 µs/token/层；只有重建 CUDA 上下文（重启容器）才恢复。
   反证未清：外部探针健康（但它无法按引擎的量级分配——引擎跑满 VRAM，没有空闲可分配），所以无法直接证实。

## 9.10.11 实操（均已验证）
- 任何 ≥64K prompt 之后**重启容器**；或把长 prompt 放到新会话的第一件事。
- chunk 保持 2048；prefetch 保持 16384（关掉不修中毒，且干净态慢 24%）。
- `gpu_memory_utilization` 不低于 **0.93**（0.92 及以下因 KV 不足直接起不来）。
- `max-model-len` 不要开得比需要的更大（它按比例放大 QSA/元数据预留）。

## 9.10.12 下一步候选 A/B（各 ~6 分钟重启，只改一个变量）
1. `ple_ssd_cache_mb` 512 → 4096（宿主 RAM 充裕）：测“行缓存抖动”是否是载体。
2. `--max-num-batched-tokens` 2048 → 1024/512：看中毒阈值是否随 chunk 形状移动。
3. **强制块分配顺序**：中毒后先在同一个容器里反复申请/释放短请求（或先跑一个短序列）把低编号块“用旧”，再看短请求是否恢复——用来验证 §9.10.4 里唯一的物理差别（块编号 2~13 ↔ 42~50）是否就是载体。

# 9.11 第四次咨询 gpt-6-astra（2026-09-29 09:5x）：纠正 + 唯一判别实验

## 9.11.1 Astra 接受的/纠正的点
1. **“CUDA event 间隔 = 30 ms”不等于“内核在 SM 上执行了 30 ms”**：区间还可能包含提交空档、被抢占、依赖等待。满时钟 + 低功率与“内存停顿”一致，但与“断续执行”也一致。→ 我不能宣称“已证明是 SM 停顿”。
2. **“只有 gather 路径被打”说得太满**：总增量 1933.4 ms 中，QSA 注意力只占 333.6 ms（**17%**），GDN 模块占 1166.5 ms（**60%**）；而 `GDN_FUSED`（核心核）不变 ⇒ GDN 的增量大多在**模块内的投影/GEMM** 上，单纯“QSA KV 地址坏”解释不了 GEMM 变慢。
   我的反证：`stage("GDN")` 的模块级事件区间会**跨图段（eager break）**，把宿主/PLE 空档包进去，属于已知的区间假象（§9.6 的“module interval straddle”）；窄口径的 `GDN_FUSED` 才是可信的内核级数据。**这一点待用 §9.11.2 的实验一并验明。**
3. **block id 是 CUDA 分配内的虚拟地址**，不是物理页；`align` 助手只是从各组的 block table 取 `(seq_len-1)//mamba_block_size` 起的条目，**并没有**对 QSA block id 取模 8 ⇒ “8 块组 = 8 槽冲突”没有依据。
4. **prefix cache reset 不会恢复低编号块**（只清 hash，不重建 free-block 队列）⇒ “刷前缀缓存把块号压回低位”这个廉价 H3 测试是无效的/被混淆的。
5. 排除：持续性降频/降功率（已由 10 Hz 采样否掉）、全局算力/HBM 带宽损失、PLE SSD 服务时间为主因、中毒后短请求“活变多”、以及**稀疏注意力内核里的自旋循环**（该内核没有 inter-CTA 轮询；TOPK 是另一个核且未膨胀）。
6. 96K 触发吞吐不变是反对“全局慢模式”的强证据；但**聚合的 50~64 s 会掩盖局部惩罚**，除非逐 chunk 计时。

## 9.11.2 Astra 给的唯一判别实验：**冻结输入的 QSA「位置 × 启动方式」四格交叉**
在**同一层 QSA、同一个 fresh 2048-token 请求、干净态与中毒态各做一次**，用**引擎自己的 CUDA 上下文**（外部上下文答不了地址相关问题）。四格：

| | 原始 K/V 分配 | 把被引用的 K/V 块拷到 scratch（私有 block table，只换指针/表） |
|---|---:|---:|
| 单独 eager 启动 | T_OE | T_RE |
| 单核 CUDA graph 重放 | T_OG | T_RG |

要点（Astra 原文摘要）：
- scratch 只需覆盖**一个 2048-token 请求在同一层引用到的 K/V 块**：bf16、head_dim 256 时 `2×2048×H_KV×256×2 B = 2 MiB × H_KV`（4 个 KV head ≈ 8 MiB，不含 padding），**不需要复制整个 KV 池**。
- 先测原始地址，**再**拷贝（拷贝会暖缓存/改常驻，污染结果）；拷贝要按**整个存储块**、保持布局/stride，并用私有 block table 映射同样的逻辑块。
- 用**同一个已编译内核**、同样的 grid/specialization/stride/count/scale/Q/选择/gate/输出，只改 K/V 指针与表；不要走会重选 specialization 的 wrapper。
- 两种地址都在**同一条诊断流**上先 eager、再用“start event → kernel → end event”的单核 graph 测；每格 3 次、交替顺序、首次单独记录；编译与建图不计时。
- 总代价：中毒态 12 次 × 30 ms ≈ 0.36 s 内核时间/状态，加准备时间。

判读矩阵（Astra）：
| 中毒态结果 | 支持 |
|---|---|
| 原始 ≈30 ms、迁移后 ≈2~4 ms（两种启动方式都是） | K/V 位置/常驻/访存局部性因果（比值 >4× 且换序复现即可定案） |
| 两种地址 eager 都 ≈30 ms、单核 graph 都 ≈2~4 ms | 病灶在 **eager 提交/时序空档**，与地址无关 |
| 两种隔离方式都快、只有生产调用 ≈30 ms | 病灶在**周边执行序列**（入边依赖/干扰/首次触碰） |
| 原始地址立即重复就变快 | 时间性暖机/常驻起作用，不能把后续快归因于地址 |
| 四格都 ≈30 ms | 迁 K/V 与去掉周边启动链都不够；其他操作数、上下文调度、内核内部行为仍待查 |

## 9.11.3 Astra 建议同时抓的状态（对照干净/中毒各一次/层）
- 真实 Q/K/V/输出/gate/选择数组的**指针**、storage base、offset、shape、byte stride、dtype（验“同一个内核”是否真是同一套存储关系）。
- **完整**的 block-table 条目（按组）——min/max/jumps 汇总会掩盖不同映射。
- 每行有效计数与选择下标数组的**逐元素**比较（相等 max/sum ≠ 相等访问序列）。
- GDN 的 state 下标、`has_initial_state`、seq offset、prefill/decode 计数。
- graph/segment 身份、stream 身份、producer/wait/record 关系。
- **裸内核调用本身的宿主耗时**（若一次 launch 调用就花到 30 ms，事件区间要重新解释）。
- 读回一律放在计时**之后**（`.item()`/`.cpu()` 会排空队列、扰动状态）。
- 若要内核内遥测：每 CTA 入口/出口 `%globaltimer` + SM id，写进唯一索引槽（无原子，约 20N 字节），比较
  `T_CTA envelope = max(exit) − min(entry)` 与事件区间：区间≈30 ms 而 CTA envelope≈2 ms ⇒ 空档在 CTA 执行之外；两者都≈30 ms 且每 CTA 生命周期变长 ⇒ 延迟发生在 CTA in-flight 期间。（注意：`%globaltimer`/`clock64()` 都不是活跃周期计数器，且含被抢占时间。）

## 9.12 交叉实验（Astra 4 格 + 扩展）实测：中毒态下**同一个内核自身执行变慢**，与地址/布局/启动方式/数据取值均无关

### 9.12.1 探针实现（`/home/hong/vllm/qsa_ops_instr.py`，挂载为 `ops/qsa.py`）
在 `qsa_sparse_paged_attention` 生产启动之后插入探针：用**本次调用自己的** Q/K/V/选表/块表/输出操作数，
在私有 stream 上（`s.wait_stream(current)`；每 rep 前后 `s.synchronize()`，引擎线程被阻塞 ⇒ GPU 排空、可隔离计时）
以 CUDA event 计 3 rep。单元（arms）：

| 记号 | 含义 |
|---|---|
| `OE` | 原始地址 + 生产选择表，eager 启动 |
| `RE` | 把**被引用的整块** K/V 复制到全新 scratch（保持 layout/stride，私有 block table），eager |
| `SH` | 同一 K/V 存储、block table 整体平移 `+7 mod nblk`（纯地址扰动，写 scratch 输出） |
| `OG`/`RG` | 原始 / 迁移地址，**单内核 CUDA graph** replay |
| `OE2` | **最后**重测一次 `OE`（检验“探针自身在清状态”） |
| `RK` | scratch 里的 K/V **全填 0**（数据取值 vs 执行上下文） |
| `G4` | 4096³ bf16 matmul（~137 GFLOP）同 stream 对照（检验“整个 GPU 是否变慢”） |

关键工程点（踩过的坑，务必保留经验）：
1. **必须用本次调用的操作数**。先前版本在 PLE eager break 里用“快照”指针重放，快照指向的缓冲被后续步骤复用 ⇒
   读到 `bt_uniq=1`/全 0 的**陈旧内存**，测出 2.0→13.5 ms 的假信号（已作废）。
2. **弹夹式开火（arming file）**：探针平时**完全关闭**（`/tmp/qsa_probe_arm` 不存在即 0），只在我需要时写入
   stride（内容=每 N 次合格调用探一次）。这样**触发中毒时零扰动**（此前常开探针会让中毒根本不出现）。
3. 合格调用判据：`num_splits==1 && q.shape[0]>=1024 && 非捕获 && BreakableCUDAGraphCapture.current() is None`
   （即**replay** 期）。用 py-spy 证实 `qsa_sparse_paged_attention` 在 replay 期确实由 Python 执行
   （栈：`_xlaunch → qsa_sparse_paged_attention → forward_qsa → _run_qsa → bcg.replay`），
   而库内核段则只 replay 不跑 Python。`ops/qsa.py` 位于 `/opt/vllm/src` 且是活的模块（非 venv 副本）。
4. `_xp_gate` 每 2 s 打印真实调用画像（`nq/prefill/splits/cap/active/bt0/cnt_max/…`），避免“静默门”再次发生。

### 9.12.2 事实（本机 GPU0，`--max-num-batched-tokens 2048`，每次请求全新 token id）
**干净态**（容器 `1105c74c761d` 前、`18fa…`）：fresh 2048 = 0.84~0.96 s；8192 = 2.32 s。
探针（nq=2048, `sel_w=2051`, `cnt_max=2048`, `cnt_sum=2098176`, `ti_max=2047`）：
`OE=2.17~2.67 ms`、`RE=2.10~2.61`、`SH=2.14~2.73`、`OG=2.10~2.69`、`RG=2.06~2.60` ⇒ **四格全等**，
且 `OE≈2.2 ms` 与此前图内事件测得的干净 2.2 ms 一致 ⇒ 探针方法可信。

**中毒态**（2019: fresh 2048 = 2.027 s / 1010 tok/s，8192 = 9.72 s；本轮再现两次，见 9.12.3）
- 全毒时刻（nq=2048，**工作画像与干净态完全相同**）：`OE=21.76/21.89/21.69`、`RE=22.13/21.71/21.84`、
  `SH=22.59/21.96/21.60`、`OG=21.80/21.52/21.74`、`RG=14.07/13.96/14.23` ms ⇒ **≈10× 慢，四格全慢**
  （Astra 矩阵第 5 行：迁移与去启动链都不够）。
- `host_ms`（裸 launch 入队耗时）= 0.03~0.4 ms（RE 有一次 2.63 ms 冷启动）⇒ **宿主提交成本不是原因**。
- 半恢复时刻同一次探针内：`OE=7.0` 而 `OG=2.11 / RG=1.99` ⇒ 此时 graph 比 eager 快 3.4×，且**同一 arm 三 rep 稳定**
  （⇒ 不是“首次触碰”，而是当时状态本身在变：探针按顺序推进时状态在被清）。
- 中毒态 8192 请求的 chunk（nq=1584/1856, `cnt_sum=3.25e6/3.80e6`, `ti_max=6335/8191`）：
  `G4=0.70~0.74 ms`（**190 TFLOPS，同 stream 同一时刻满速**）而
  `OE=3.34/3.82/3.89/3.93`、`RE=3.27/3.70/3.76/3.77`、`SH=3.33/3.50/3.87/3.93`、
  `OG=3.34/3.77/3.78/3.82`、`RG=3.28/3.69/3.70/3.73`、`OE2=3.35~3.78`、**`RK（K/V 全 0）=3.12/3.66/3.67/3.69`**
  ⇒ 同一内核、同一存储关系、同样 0 填充数据，**快慢完全一样**。

### 9.12.3 新确立/推翻的结论
- **确立**：中毒态下 `qsa_sparse_paged_attention` 的**自身 GPU 执行时间**从 2.2 ms 涨到 ~22 ms（同一工作画像、
  同一编译内核、同一网格/特化）；该结论由**图外独立方法**复现了此前图内事件的 2.2→30 ms。
- **推翻（H1/H3 变体）**：K/V **物理位置/常驻**（整块迁移到全新 198 MiB scratch，私有紧凑 block table）**无影响**；
  block-table 取值（整体平移 7 个块）**无影响**；K/V **数据取值**（全 0）**无影响**。
- **推翻（eager 提交假说）**：单内核 CUDA graph replay 与 eager launch **同速**（全毒时刻 21.8 vs 21.8 ms）；
  宿主 launch 入队耗时 0.03~0.4 ms，量级完全不够解释 20 ms。
- **推翻（“整个 GPU/上下文被抢占或降频”）**：同一 stream、同一时刻 4096³ GEMM = **0.70~0.74 ms（~190 TFLOPS）**，
  即 SM 完全可用；且此前测得中毒态 SM 1477 MHz（满频）、105 W、无 throttle 标记。
- **新事实（可逆！）**：中毒态**不是**“只能靠重启清除”——探针活动（私有 stream + 事件 + 逐 rep 同步排空）
  会**逐步清除**它：arming 期间 2.805 → 1.404 s，8192 6.03 → 随后 fresh 2048 实测 0.89 s（完全干净）。
  这也解释了为什么**常开探针的容器根本不中毒**（观测者效应）。
- **新事实（可复现性有状态依赖）**：同一容器内**首次** 98304 触发必中毒（本轮 3/3），之后再次 98304 触发**不再中毒**
  （0.684 s 干净）⇒ 触发条件与**容器启动后的分配/池状态**有关，不只是 prompt 长度。
- 与既有结论一致：decode 不受影响（21.1 ms/token）；PLE SSD `io` 不变；512 尺寸也会按每 token 比例变慢。

### 9.12.4 仍然开的岔口
在“同内核 + 同操作数 + 同地址关系 + 同启动方式 + 同数据取值 + 设备满速”全被排除后，剩下的两个方向：
1. **内核态本身在变**（例如实际加载的 cubin/寄存器数/占用率/`maxnreg`、L1/shared carveout、`cluster` 配置等
   在两次调用间不同）——需要抓 `cudaFuncGetAttributes`/`cuOccupancyMaxActiveBlocksPerMultiprocessor`、
   cubin hash、每 CTA 的 `%globaltimer` 包络。
2. **state 是“可被排空”的事件/队列副作用**（探针同步即清除它），即某种 *in-flight* 的引擎侧 GPU 工作
   与 prefill 内核串扰但**不拖慢 4096³ GEMM**（⇒ 只对特定资源（例如 gather/共享内存/`__ldg` 路径、
   或 atomic/barrier 行为）敏感）。
下一步实验优先：在内核里写 per-CTA `%globaltimer` 入口/出口 + SM id 到唯一槽，比较
`T_CTA envelope` 与事件区间（区间 22 ms / envelope 2 ms ⇒ 空档在 CTA 之外；两者都 22 ms ⇒ CTA in-flight 期间变慢）。

### 9.12.5 第 5 次 Astra 咨询（`/tmp/astra_reply5.md`，9.2 KB）结论与下一步
Astra 判定：数据/位置/取值/启动方式/宿主提交/SM 频率/张量核吞吐都已被排除 ⇒ 病灶落在
**GPU 执行与调度边界**。机制先验排序：① 残留 GPU 工作 / stream 调度 / 排队依赖积压（最高）；
② 存储系统压力（全局访存请求队列、TLB/页表重放、L2/DRAM 分区压力、互连背压）；③ 有效 SM 常驻率下降
（并发工作/调度器干扰）；④ CUDA graph/驱动调度状态放大；⑤ WSL2/驱动 context 级调度缺陷；
⑥ 变化的 Triton 二进制或静态占用率（低）；⑦ 时钟/热/硬件（极低）。
重要提醒：**阻塞引擎线程只阻止新的提交**，并不移除已排队的 graph 工作、未完成的访存、stream 依赖或
已驻留/交错的 GPU 工作 ⇒ 我的“排空即清除”观测与①相容。4096³ GEMM 只证明**张量核吞吐可用**，
不证明 attention 能拿到正常的**访存服务**或 **CTA 常驻**（compute-bound 的 GEMM 可以在 memory-bound
attention 出现长空档的同时跑满 190 TFLOPS）。

**Astra 指定的下一个决定性实验**：给诊断内核加**每 CTA `%globaltimer` 入口/出口时间戳 + `%smid`**
（同 grid/特化/num_warps/num_stages/操作数；只由每 CTA 一个 lane 写；无逐线程 trace，无额外 barrier），
**每状态只跑一次**（不要 3 rep —— 我自己已证明重复探测会改变状态），先只能用 eager 私有 stream。
记 `T_evt`（事件区间，ns）、`D_i = end_i-start_i`、`T_first=min(start)`、`T_last=max(end)`、`T_env=T_last-T_first`、
`U`=所有 CTA 区间并集长度、`head_tail=T_evt-T_env`、`T_env-U`、median/p95 `D_i`，并把每个 CTA 的
`x_start_i=100*(start_i-T_first)/T_env`、`x_end_i` 列表比较（按 CTA id 和按 start 时间各排一次）。

判读矩阵（Astra 原表要点）：
| 中毒态观测 | 结论 |
|---|---|
| `T_evt`×10，但 median/p95 `D_i` 仅 1.0~1.5× | 单个 CTA 执行正常，时间丢在 CTA 之外 |
| `T_evt`×10，`T_env` 接近干净，`head_tail` 占大头 | 首 CTA 前/末 CTA 后的延迟 ⇒ stream 调度/排队依赖/发射准入/尾部同步 |
| `T_evt`×10，`T_env` 也涨，`D_i` 近干净，`T_env-U` 大 | CTA 速度正常但**有长空档** ⇒ 有效常驻率下降/调度饥饿/与其他 GPU 工作交错 |
| `T_evt`×10，median/p95 `D_i` 同比×10，重叠模式不变 | **CTA 在飞行中被拖慢** ⇒ 存储系统压力/翻译重放/特定 SM 侧停滞 |
| 仅特定 `%smid` 或特定 wave 的 `D_i` 变大 | 分区/SM 局部或并发工作干扰（比全局变慢更强的证据） |
| `D_i` 分成两簇（一簇近干净、一簇很慢） | 间歇性干扰/wave 级重叠/SM 局部竞争 ⇒ 与 `%smid`、start 时间对照 |
| `T_evt` 与 `T_env` 都大，但 `U≈T_env` | CTA 一直在跑 ⇒ 问题在 CTA 内部执行 |

对本机当前数字，两类假设的预测不同：**CTA 之外延迟** ⇒ `D_i` 仍 2~3 ms（归一化包络出现大片空白/重叠 CTA 数骤减）；
**CTA 内变慢** ⇒ `D_i` 本身升到 15~22 ms 且 CTA 在区间内几乎连续活跃。若确认是 CTA 内变慢，
再加 4 个阶段标记（index/table、K/V load+selection、attention 算术、输出写出），比较 `bad/clean` 比值：
只有 load/selection 的比值大 ⇒ 访存/翻译压力；各阶段接近均匀 ⇒ 广泛的 in-flight 调度/资源问题；
集中在 barrier/final store ⇒ 同步/完成行为。另外顺手比对两次运行的 cubin hash 与静态属性
（寄存器/共享内存/warp/stage）——Astra 认为二进制变化概率最低，但比对成本极低。

## §9.13 决定性实验：per-CTA `%globaltimer` 追踪 + 同刻内存控制臂（Astra #5 方案）

Astra #5 的裁决是「故障在 GPU 执行/调度边界」，要求一次**每 CTA 时间戳**实验来区分
CTA-外调度 vs CTA-内飞行态。本轮在 QSA op 内实现了该方案，并在同一次探测里加了
**同刻内存服务控制臂**，最终把机理定位到「非连续/聚集类访存坍塌」。

### 9.13.1 实现（`qsa_ops_instr.py`）
- **追踪内核**：把本模块自身源码中 `_qsa_sparse_paged_gqa_splitk_kernel` 的文本
  （从 `@triton.jit` 到下一个 `@triton.jit`）复制、改名 `_qsa_trace_kernel`、在签名末
  插入 `trace_ptr`、在 `) -> None:` 后注入 prologue、在体尾注入 epilogue，写到
  `/tmp/_qsa_trace_kernel_gen.py` 再 `importlib` 导入。除 `trace_ptr` + 若干 store /
  inline-asm 读外与生产内核**逐字节同体**、同 grid/constexpr/num_warps/num_stages。
- **时间源**：`tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l,r', [tl.arange(0,1)], ...)`。
  约束串必须写成 `'=l,r'`（输出约束 + 输入约束）；只写 `'=l'` 会触发 LLVM
  `number of input constraints does not match` **UNREACHABLE 直接崩引擎**（本次踩过，
  已在一次性容器里单独验证 asm 用法后才重新上线）。`%globaltimer` 单位 ns、分辨率约 32 ns，
  值与 `time.time()*1e9` 同量级（可跨主机时间轴对齐）。
- **槽位**（stride 16 int64）：0=t0、1=t1、2=smid、3=prologue、4=index/table、
  5=K/V load 之前的窗口、6=math+剩余、7=num_tiles。
- **触发机制**：主 arming 文件 `/tmp/qsa_probe_arm`（内容=步长，只对 replay 期、`num_splits==1`、
  `q.shape[0]>=1024` 的调用生效）+ 独立文件 `/tmp/qsa_trace_arm`（控制追踪臂，便于归因）。
  探测期**必须撤防**（观测者效应已多次复现）。
- **观测者效应（重要教训）**：为了把「等待 K/V 数据到达」从「算术」里拆出来，曾注入
  `tl.sum(keys)`/`tl.sum(values)` 强制等待 → 该改动**破坏了 Triton 的软件流水**
  （num_stages=2 的 cp.async 预取），使追踪内核本身在**干净态**从 2.1 ms 变成 70 ms
  （33×）→ 相位数据整体失真（与 `OE` 对照才发现）。已回退为「只加时间戳」的忠实版本，
  该版本干净态 2.19 ms vs 生产 `OE` 2.14 ms（+2%），可信。

### 9.13.2 包络/并发（忠实追踪内核，nq=2048、同一 work profile：`cnt_max=2048 cnt_sum=2098176 ti_max=2047`）

| 指标 | 干净 | 中毒（重度时刻） | 比值 |
|---|---|---|---|
| `T_evt`（CUDA event） | 2.19 ms | 88.46 ms | 40× |
| `T_env`（首 CTA 起→末 CTA 止） | 2.184 ms | 88.068 ms | 40× |
| `U`（CTA 区间并集） | 2.184 ms | 88.068 ms | — |
| `T_env − U` | **0.000 ms** | **0.000 ms** | 两态均无空档 |
| 并发 CTA 数（40 桶剖面） | 恒定 ≈222 | 恒定 ≈222 | **相同** |
| 使用 SM 数 | 74/74 | 74/74 | 相同 |
| `D_i` 中位 / p95 / max | 190 / 347 / 389 µs | 692 / 27 699 / 61 190 µs | 3.6× / **80×** / 157× |

**判读**：两态都有「连续覆盖、零空档、恒定并发、全部 74 SM」⇒ Astra 的候选①（调度/排队空档）
与③（有效驻留降低）**被排除**；差别全在 **CTA 内部**，且是**尾部**（中位只 3.6×，p95 80×）。

### 9.13.3 同刻内存控制臂（同一次探测、同一条流、交替执行）

| 臂 | 含义 | 干净 | 中毒 | 比值 |
|---|---|---|---|---|
| `G4` | 4096³ bf16 matmul（137 GFLOP） | 0.70 ms | 0.69 ms | 1.0× |
| `BW` | 连续 32 MiB `copy_`（走 memcpy/DMA 路径） | 1394 GB/s | 1394 GB/s | 1.0× |
| `KRD` | **真实 KV 池**的稠密读（16 块 ≈52 MB） | 626 GB/s | 626 GB/s | 1.0× |
| `LAT` | 8 B 随机依赖链，296×32=9472 并发链 | 245.8 ns/次 | 403.5 ns/次 | 1.6× |
| `KSC` | **真实 KV 池**上 16384×2 KB 随机行聚集 | 117 GB/s（0.145 ms） | **1.6 GB/s（10.4 ms）** | **73×** |
| `FSC` | 同样模式、**全新** 64 MB 缓冲区 | 176 GB/s（0.095 ms） | **0.8 GB/s（19.9 ms）** | **210×** |
| `STR` | 全新缓冲区、2 KB 行 + 4 KB 步长（非随机） | 176 GB/s（0.10 ms） | **0.8 GB/s（19.9 ms）** | **200×** |
| `OE` | **生产 QSA 内核**（eager，3 次） | 2.10 ms | 22.3 ms | 10.6× |
| `OG`/`RG`/`RK`/`RE`/`SH` | 图重放 / 零填充 K/V / 重定位 K/V / 换表 | 2.0–2.2 ms | 14.3–22.7 ms | 全臂同慢 |

另外：`KSC` 的**第二次**重复比第一次更慢（10.4 → 20.8 ms），即「该模式本身会让状态继续劣化」。

### 9.13.4 结论（本轮定位）

1. 中毒态的特征是：**非连续 / 短行 / 聚集类访存（gather、行步长读、页表选块读）服务坍塌
   100–220×**，而**大块连续传输、DMA/memcpy、张量核 GEMM、高 MLP 的 8 B 随机访问几乎不受影响**。
   该结论由**同一时刻、同一内核、同一流**的对照臂给出，且两态各自成立（干净态同类模式 117–176 GB/s）。
2. 这恰好对应 QSA 注意力的访存形态（按 block_table 选块 + 每 token 256 B 散读、每 warp MLP 低）
   ⇒ 只有它慢 10.6×；MoE 稠密 GEMM 1.5×；decode 工作集小、顺序性强 ⇒ 几乎不变。
3. 尾部特征（p95 80×、max 57 ms/CTA，`KSC` 第二次更差）指向：**少数访问（页）具有巨大延迟**，
   deep-MLP 的连续流可以掩盖它，低 MLP 的散读不能。这解释了「同一 cubin、同一地址、同一数据、
   byte 相同启动参数却快慢 10×」以及「可逆、可被探测活动排空、严重度随时间漂移」。
4. 至此 Astra #5 排名中：①（队列/调度空档）✗、③（驻留/占用）✗、④（CUDA graph/驱动调度态）✗、
   ⑥（cubin/静态属性变化）✗（同 cubin 同参数）、⑦（时钟/热/硬件）✗（全时钟、无节流、GEMM 满速）；
   剩下 **②（内存系统服务）被证实**，且进一步细化为「**非连续/短传输路径 + 长尾延迟**」。

### 9.13.5 待办（下一步单变量实验）
- 逐访问延迟直方图：数百方次独立随机访问、MLP∈{1,4,64,1024}，分别报告 p50/p99/p99.9/max；
- 「少数页病态」vs「全体长尾」的判别：同一 buffer 内按页分组统计，比较固定页集合 vs 随机页集合；
- 粒度扫描：8 B / 128 B / 2 KB / 1 MB 在同一 MLP 下的有效带宽曲线（bandwidth-vs-MLP）；
- 与 WSL2/dxgkrnl 的页驻留/迁移/驱逐行为做对照（只做只读观测）。

### 9.13.6 Astra #6 的裁决与下一步（`/tmp/astra_reply6.md`，17316 B）

**裁决**：同意上述现象学结论，但措辞要更准确——这不是「DRAM 变慢」，而是
**address-sensitive, heavy-tailed completion/replay regime**（地址敏感、长尾的完成/重放态）。
理由：一次被计时的 load 除了 DRAM 延迟，还可能包含 L1/L2 miss 处理、TLB 查找与页表走查、
失败/延迟事务后的 replay、memory-fabric 排队、scoreboard 等待、驻留/迁移处理、以及取指/退休延迟。
新增要点：
- **CTA 连续覆盖 + 恒定并发 ≠ 有用进展**（warp 可以驻留但一直等内存依赖）；
- `FSC`/`STR` 也慢 ⇒ 与该分配/KV 内容无关（但不排除「少数页/映射病态」落在新分配上）；
- **`KSC` 第二次比第一次更慢**是最有信息量的单点：正常 cache miss 第二次应更快 ⇒
  说明存在**主动劣化中的子系统状态**（replay 压力 / 翻译-驻留工作 / 队列堆积），且**探针本身在加重它**；
- 9472 条并发链只退化 1.6× 与「低 MLP 长尾严重」不矛盾（全局 MLP 高可以掩盖单次等待）。

**WSL2/驱动层面排序**（Astra）：① GPU VM/TLB/页表走查/驻留/replay 态（最高）② memory-fabric/
分区调度器对稀疏事务的歧视 ③ WSL2/WDDM/dxgkrnl 的内存管理使首个大工作集进入异常驻留/映射态
④ cache/地址分区冲突 ⑤ ECC/压缩/静态 carve-out（最不可能，且无法解释「状态随分钟漂移」）。
另建议一个高价值对照：**被动流逝时间 vs 主动探针流量**（若只有流量能恢复，则「后台自行排空」的说法不成立，
应改称「探针改变了驻留/翻译/cache/队列状态」）。

**Astra 指定的下一个实验（单一温度计）**：
- 用 `%globaltimer` 夹住**单次显式 load**（用 inline asm 保证不被消除，结果必须 live），
  **逐样本**记录 `dt`、`%clock64` 增量、`%smid`、地址/页标签（4 KB/64 KB/2 MB 三档）、arm、MLP、tile 序号；
- 样本量 25 万–100 万，日志分箱（<0.5 µs … >4 ms），报告 p50/p90/p99/p99.9/p99.99/max 与阈值存活率；
- **必须按时间分 tile**（如 64 tile × 16384 样本）记录，因为探针会改变状态（不能假设平稳）；
- MLP 定义为**每 SM 在途逻辑 load 数**，取 {1, 4, 64, 1024}，同一内核内随机化 arm 顺序；
- 页归属统计 `q_i(T)=P(dt>T | page i)`，固定页集合、跑 16+ 遍，用于区分「少数病态页」vs「全体长尾」；
  判定表：同一小部分页恒慢 ⇒ 页/映射病态；均匀长尾 ⇒ 全局地址敏感分布；仅某 allocation 慢 ⇒ 分配/映射问题；
  pass2 普遍变慢 ⇒ 探针诱导的全局劣化；
- 尺寸依赖：8 B / 128 B / 2 KB / 1 MB（同一线程数与指令序列、固定字节数或固定区域数），
  并对 128 B 子请求单独计时，避免把「1 MB 请求时长」直接与 8 B 比较；步长扫 4 KB/8 KB/16 KB/32 KB/64 KB/2 MB；
- bandwidth-vs-MLP 曲线可区分：**latency-only**（高 MLP 追平 ⇒ 只是长尾被掩盖）、
  **replay/事务容量**（MLP 1024 也上不去）、**翻译/驻留**（依赖页工作集与步长阈值）、**SM 局部**（各 SM 曲线不同）。

**Astra 给出的反证判据（下轮据此判定）**：若中毒态下**新缓冲区单次 load 的延迟分布在 MLP=1 时
与干净态接近（p99.9/p99.99/毫秒级尾部都在 ~20–30% 内）**，而 2 KB 生产形态内核仍慢 10×，
则「罕见的巨大访存延迟」**不是充分解释**，下一步应转向 QSA 专有的 warp replay、取指发射、
地址生成串行化、同步、或软件流水交互。另一反证：若中毒态尾部事件几乎都落在**同一小批页 ID** 上，
则模型应改为「少数病态页/映射」而非「整条随机访问通路长尾」。

**当前已有的粗粒度对应**（本报告 §9.13.3 可直接当「单温度计」的第一版）：
`LAT`（每 lane 单链 = 每 warp MLP=1，9472 并发链）中毒态仅 1.6×（245.8→403.5 ns），
而 `FSC`/`STR`/`KSC`（16 行/次的大在途模式）坍塌 73–210× ⇒ 指向**在途请求容量/重放**而非
单纯单次延迟；下一步需按 Astra 设计在**同一内核**内做 MLP 曲线与页标签统计来确认。

## §9.14 逐访问延迟温度计（Astra #6 指定实验）：结果**否证**了纯延迟机制

### 9.14.1 仪器与踩坑记录（重要，供后续复用）
- **`%globaltimer` 在本机分辨率是 1024 ns**（实测 20000 次连续读数只有 {0, 1024} 两个值）
  ⇒ 亚微秒访存计时必须用 **`%clock64`**（实测最小步进 31 cycle ≈ 21 ns）。
- **计时读必须被 load 的数据依赖门控**：普通 S2R 与 load 无依赖，测到的是「发射」而非「完成」。
  可用 PTX 谓词门控解决（本实现）：
  ```
  { .reg .pred %pp; .reg .b64 %vv;
    mov.u64 $0, %clock64;
    ld.global.u64 %vv, [$3];
    setp.ne.u64 %pp, %vv, 0;
    @%pp  mov.u64 $2, %clock64;
    @!%pp mov.u64 $2, %clock64;
    mov.u64 $1, %vv; }
  ```
  三步输出（t0, value, t1）与 `tl.inline_asm_elementwise(..., dtype=(tl.int64,)*3)` 配套。
- **零填充缓冲区会被 L2 压缩**，测得 dt≈0（假信号）⇒ 控制缓冲区必须填**不可压缩随机数据**。
- 每样本槽位必须含 `pid`（否则 74 个 CTA 互相覆写，表现为「98% 样本为 0」）。
- perm 数组的**索引模数必须是 perm 长度**，不能用缓冲区元素数（本次因此越界读 perm → 非法访问）。
- **KV 池是不可连续寻址的切片**（`stride(0)=21745152` 而单块只有 1622016 个元素），
  且是大块（可能是 VMM/稀疏映射）分配 ⇒ 只能在**每个 block 自身的连续段**内取址：
  `off = b*stride0 + randint(0, nrow*stride1)`（本实现）。任何一次越界都会杀死整个引擎进程
  （`cudaErrorIllegalAddress`）。
- 载体地址（int64 词偏移）必须配上 **int64 视角的张量**做 `base_ptr`，否则 Triton 会按元素类型缩放。

### 9.14.2 结果：单次访问**完成**延迟（`%clock64`，1M 样本/态，round 0 = 最未排空时刻）

| 每 warp 在途请求数（每 SM） | 干净（同容器 r=1） | **中毒（r=0）** | 比值 |
|---|---|---|---|
| 1（74）fresh | 0.168 µs | **0.340 µs** | 2.0× |
| 4（296）fresh | 0.294 | 0.416 | 1.4× |
| 32（2368）fresh | 0.483 | 0.537 | 1.1× |
| 512 fresh | 1.156 | 1.288 | 1.1× |
| 1024 fresh | 1.933 | 2.030 | **1.05×** |
| 1（74）kvpool | 0.166 | **0.441 µs** | 2.7× |
| 4 kvpool | 0.272 | 0.432 | 1.6× |
| 32 kvpool | 0.465 | 0.578 | 1.2× |
| 512 kvpool | 1.437 | 1.702 | 1.2× |
| 1024 kvpool | 2.548 | 2.675 | **1.05×** |

中毒态所有臂 **`n>10µs = 0`**、max ≤ 5.5 µs，且 **MLP 曲线形状与干净态完全一致**。
同刻其它量：`G4=0.69` / `BW=1394 GB/s` / `KRD=626 GB/s`（1.0×），
`LAT(ns/load)=698 vs 干净 245`（2.8×），而 **`KSC=1.6`、`FSC=0.8`、`STR=0.8 GB/s`（坍塌 100–220×）**；
追踪内核：`t_env=89.9 ms`、`D[med/p95/max]=675/31752/66722 µs`、相位 `pr/ix/kv/math/ep=1/1/103/544/1 µs`。

### 9.14.3 结论（本轮的转折）

1. **Astra 给出的反证判据成立** ⇒「罕见的巨大单次访存延迟」**不是充分解释**：
   中毒态下单次访问完成延迟只差 **1.05–2.7×**，无微秒级以上事件，
   而同一时刻 2 KB 行聚集内核的**吞吐**坍塌 **100–220×**，QSA 内核慢 10×、其 CTA p95 达 31.8 ms。
2. ⇒ 差别不在「单笔访问多慢」，而在**大量短事务/大网格内核的推进能力**：
   延迟正常、并发正常（追踪显示并发恒 222、无空档），但吞吐不成比例地崩塌，
   且**少数 CTA 的整段执行被毁**（中位仅 1.9×，p95 47×，max 99×）。
3. 与所有既有观测自洽的读法：中毒态损害的是 **warp/CTA 的推进（replay、发射、网格补充）**，
   而不是访存延迟本身。这解释了「同 cubin、同地址、同数据、byte 相同启动参数」；
   也解释了为什么 **74 CTA 的温度计正常（1.05–2.7×）**、**单次 memcpy 正常**、
   **稠密 GEMM 正常**，而 **4096 CTA 的 QSA 内核**、**大网格的 PyTorch 聚集/步长内核**、
   **大网格 MoE（1.5×）** 受影响。
4. Astra #6 预设的第二条分支（“rare enormous latency 不充分时转向 QSA 专有 warp replay、
   指令发射、地址生成串行化、同步、软件流水交互”）成为当前主假设。

### 9.14.4 下一个实验（单变量、低成本）：**网格规模扫描**
同一温度计内核，仅改 `grid`（74 / 592 / 4736 / 37888 CTA，各保持总样本≈5×10^4、
每 CTA 迭代数相应减少），同时保留 `KG=1/32` 两档，在干净与中毒两态各跑一次。
- 若 **74 CTA 正常而 37888 CTA 坍塌** ⇒ 证实「大网格 CTA 调度/补充态」是中毒载体，
  与 QSA(4096 CTA)、大网格 PyTorch 内核被击中、而小网格/单 memcpy/稠密 GEMM 幸免完全一致；
- 若各网格规模都一样 ⇒ 载体是「短事务数量/在途请求数」本身（transaction-capacity），
  需要再对比同一网格下的 dense vs strided 变体；
- 若只有 QSA 内核慢而任何独立内核都正常 ⇒ 载体在 QSA 内核内部（replay/同步/流水），
  转入内核内逐 warp 的段标记与 `%clock64` 计数。

## §9.15 逐访问温度计的否证 + 三项新证据（WSL2 侧 / 驱逐环 / 新进程对照）

### 9.15.1 §9.14 的"长尾延迟"机制被否证
温度计（`%clock64`、谓词门控在 load 数据依赖上）在中毒态 **`n>10µs = 0`、max ≤ 5.5 µs**，
干净/中毒比值 1.05×（MLP 1024）～2.7×（MLP 1），MLP 曲线形状两态一致；
而同一时刻 2 KB 行聚集内核吞吐坍塌 100–220×、QSA 内核慢 10.6×、其 CTA `p95 = 31.8 ms`。
⇒ 「罕见巨大单次访存延迟」**不是充分解释**（Astra #6 预设的反证成立）。
**该温度计自身的缺陷**：perm 只覆盖约 1 MiB 地址足迹，因此**结构上不可能**暴露页表/TLB 类问题
——这点在解读时必须记住（§9.15.4 与此直接相关）。

### 9.15.2 重大混淆因素：一个跑了 3 小时 41 分的失控监测循环（已处置）
- 宿主 WSL 内 PID 1185000（起始 09:58:32）是一个**未被杀掉的采样循环**：
  `while :; do nvidia-smi --query-gpu=clocks.current.sm,... >> /tmp/clk_pois.csv; done`
  —— 它的清理 `kill` 从未执行，于是**无限重生 `nvidia-smi`**；每个 `nvidia-smi`
  在 WSL 里卡在 `D`（不可中断）状态，以 **≈850 次/秒**刷
  `misc dxg: dxgk: dxgkio_escape: Ioctl failed: -22`（EINVAL），共写入 186,235 行。
- 后果：**dmesg 环形缓冲（4095 条）被完全占满**，`dmesg | wc -l = 4095` 且 100% 是这条
  ——**任何真正的 GPU 故障/迁移/重置消息都会被冲掉**。这是长期的观测性灾难。
- 已彻底清除（`kill -9` 循环 + `pkill -x nvidia-smi`）：**新增消息 0 条/5 s**（原 ~35 条/5 s 完成、
  ~850 ioctl/s）。
- **但中毒完全照旧复现**（无风暴时 baseline 2048 = 1.023 s → trigger 98304 = 56.1 s →
  post 2048 = **2.292 s / 893 tok/s**）⇒ 该风暴是**混淆因素而非病因**。
- 副产品（新证据）：**中毒态下内核日志完全干净——0 条新增消息，无 Xid、无 TDR、无 reset、
  无 paging/migration 记录** ⇒ 中毒**不经过任何驱动错误路径**。
- 教训：以后跑 `nvidia-smi --query-gpu=...` 采样必须挂在 `timeout` 下，且**测完立刻验证进程数=0**；
  并且要用"新增消息数"而不是 `dmesg -w | wc -l` 来测速率（后者会把满缓冲重复计数）。

### 9.15.3 Windows 侧真 NVML 可用了，一次性排除一整族假设
WSL 里的 `nvidia-smi` 是残废版（`-q` 直接报错），但 Windows 侧的
`/mnt/c/Windows/System32/nvidia-smi.exe` 可用，实测：
- ECC：`ECC Mode = N/A`，SRAM/DRAM Correctable/Uncorrectable **全 N/A**（此卡无 ECC）；
- `Remapped Rows = N/A`、`Remapped Banks = N/A`、`Channel/TPC Repair Pending = No`；
- `HW Slowdown / HW Thermal / HW Power Brake / SW Thermal = Not Active`，Performance State `P0`；
  `Clocks Event Reasons` 全 0；
- 内存 `Used 61657 MiB / Total 65536 MiB`（与容器内一致）。
⇒ **ECC 刷洗、行重映射（坏行替换）、热/功耗刹车、降频**这一整族**全部排除**。

### 9.15.4 网格规模扫描：网格/CTA 数量也不是判据
中毒态下（**确认仍中毒**：扫描后测得 2048 = 2.817 s / 727 tok/s）、在新进程里、
用不可压缩载荷（避免 L2 压缩造假）：
- 256 B 行 @ 4 KB 步长，总字节固定，`grid = 16/64/256/1024/4096/16384/65536`：
  **46–145 GB/s，每一档都正常**（dense 对照 52–145 GB/s）；
  末端的循环重复 3 次无退化。
⇒ 上一轮"大网格 CTA 调度/补充态"的假设**不成立**。

### 9.15.5 决定性对照：同样的模式在**引擎外的新进程**里完全健康
在**同一容器、同一中毒瞬间**，用一个新 Python 进程原样复刻引擎里那条坍塌的模式
（`out.copy_(src[随机索引])`，256 B 行）：
- **0.10–0.20 ms / 4 MiB 一趟 = 25–45 GB/s**（引擎内同一模式 0.8 GB/s ⇒ 差 30–50×）；
- **连续 60 次重复：min 0.092 / med 0.115 / p90 0.142 / max 0.183 ms，>4×min 的离群值 = 0**。
- 地址足迹扫描（16 MiB → 3 GiB，即 4 096 → 786 432 个 4 KB 页）：
  scatter 22–28 GB/s **全程平坦**，dense 34–56 GB/s（**TLB 足迹到 3 GiB 不构成影响**；
  但引擎进程映射约 61 GiB，是这里最大值的 20 倍，未覆盖）。

⇒ 至此被**逐一排除**：访问模式本身、设备/驱动全局状态、罕见的毫秒级停顿、
网格规模、持续时间/字节数、地址足迹（≤3 GiB）、数据值、放置、启动参数、时钟/功耗、
ECC/坏行、任何驱动错误路径。
**剩下的唯一载体是"引擎进程自身（per-context）的状态"**——而同一进程内的
`BW`（连续 32 MiB，0.05 ms）与 `KRD`（KV 池稠密，626 GB/s）在同一瞬间仍是满速。

### 9.15.6 新定量证据：中毒状态**随活动加重**，不是被排空
post-trigger 2048 连续测量：**2.292 s → 2.817 s → 4.002 s（512 tok/s）**，
期间只做了 GPU 探针与外部测试。与"探针活动会排空"的早期观察相反
（早期 2.805 → 1.404 s），说明存在**方向相反的两种作用**，需要单变量区分。

### 9.15.7 待 Astra #7 判定的下一个实验（无需重启，可在新进程内做）
用 CUDA VMM 预留 32–48 GiB 虚拟地址，把**同一块 64 MiB 物理内存反复映射（别名）**
铺满该 VA，再让同一 scatter 模式落在**已映射页**上：
- 若坍塌 ⇒ 页表/TLB 覆盖（per-context page table）是载体；
- 若正常 ⇒ 页表排除，转向 dxgkrnl residency/UVMM 记账、或 per-context 在途容量等。

## §9.16 Astra #7 判定与规定的下一轮实验（最后一批重启内实验）

### 9.16.1 判定要点
1. **载体在"进程/CUDA context 作用域的虚拟内存层"**（WSL2 GPU-PV 路径），
   与损坏的 DRAM、全局 L2、SM 降频、普通内核调度**不相容**。
2. 指出 §9.15.5 的"新进程健康"**同时**新建了进程记录与 CUDA context，
   因此**还无法区分**「context 作用域资源」与「dxgkrnl 的 per-process 分配/residency 台账」
   —— 这正是下一步要拆开的变量。
3. 活跃候选（按 Astra 排序）：① context 页表/TLB/页走缓存 ② dxgkrnl/WDDM 的
   residency 与映射台账 ③ **中间层页表占用**（可能是某一级的有限结构而非总字节）
   ④ per-context 的 replay/页走/缺页容量 ⑤ 物理页放置/分区着色
   ⑥ VA 相关的缓存/请求/翻译分区（注意：Astra 提醒"VA 选 DRAM 分区"只能当实验假设，
   不能当架构事实）⑦ CUDA 分配器与映射拓扑（expandable segments / cudagraph 映射）
   ⑧ process-scoped WSL2 宿主状态 ⑨ pinned-host/BAR 路径（概率低，与设备侧证据不符）。
4. 算量级：61 GiB 在 4 KiB 页下 ≈ 1 599 万页、3 GiB ≈ 78.6 万页；2 MiB 页下分别是
   31 232 与 1 536 个叶表项 —— 比值足够大，能解释"阈值"型行为。
5. 对"活性加重"的解释（自增强）：每次散列访问触碰新的/弱驻留页 → 消耗有限的
   walk/replay 资源 → 停滞请求仍占着同一 CTA → 新请求继续争抢 → 结构被持续搅动 →
   少数 CTA 累积出极长尾。这与"并发恒 222/222、但 p95/max 达数十毫秒"完全一致；
   也解释了为什么 1 MiB 足迹的温度计看不到它、而 2.8× 与 73–200× 能并存。
6. 稠密访问仍快的理由：相邻扇区摊薄翻译开销、可合并/预取、MLP 高、
   不跨无关映射反复跳转、能把单次翻译延迟藏在其它有用工作后面。

### 9.16.2 它把别名实验升级为 **4 格交叉**（下一步主体）
同一进程内，用 CUDA VMM 造 **两块等大不可压缩物理块 `P0/P1`** 与
**两个等粒度但 VA 高位/对齐不同的保留区 `V0/V1`**，跑 4 种组合：

| 映射 | 判读 |
|---|---|
| `P0→V0` | 基线 |
| `P0→V1` | 与 P0 相同、VA 不同 |
| `P1→V0` | 与 V0 相同、物理不同 |
| `P1→V1` | —— |

- 慢跟着 **V** ⇒ VA 索引的翻译/页表级/TLB/VA 分区；
- 慢跟着 **P** ⇒ 物理放置/residency/分配级台账；
- 四格都慢 ⇒ **context 级**的 replay/翻译压力或 process-scoped dxgkrnl 状态；
- 只有一格慢 ⇒ 映射历史与驻留的交互；
- **第一遍快、后面变慢 ⇒ replay/residency 搅动或自增强的页走状态**。
反面结果**不能**单独否决页表假设（唯一物理页的 residency 元数据、页级重映射状态、
物理放置、分配级台账、与真实 KV 分配不同的映射粒度都仍开放）。

必须遵守的实验规范（Astra 明确要求）：先用 `cuMemGetAllocationGranularity` 取最小粒度；
别名放在**非均匀 VA 偏移**并留空洞，每个映射都要真正触碰；`reserve-only` 只作对照；
各 arm 的 **load 指令数与行数保持一致**；不可压缩载荷；乱序别名；输出可观测（防 DCE）；
**低 MLP 与高 MLP 分开测**；记录 event 时间、`%clock64` p50/p95/99/max、CTA p50/95/99/max；
**不要用总字节数当主指标**，用相同的"行操作数"。

### 9.16.3 两个更有价值的补充判据（都不贵）
1. **同进程"释放后可恢复"测试**：在中毒的引擎进程里 `torch.cuda.empty_cache()`
   （把缓存块还给驱动）+ `gc.collect()` + 同步，然后重跑 `FSC/STR` 与 `OE`。
   - 恢复 ⇒ 分配/residency 状态（池子）是载体；
   - 不恢复、只有销毁进程/context 才恢复 ⇒ context 作用域状态或不可恢复的 dxgkrnl 台账缺陷。
2. **同进程"第二 context"测试**：若能在中毒进程内再建一个 CUDA context 并仍慢 ⇒
   process-scoped（dxgkrnl）；若新的 context 快 ⇒ context-scoped。这是唯一能拆开
   "进程 vs context"的直测（第二轮方案里可以只做"近似版"：子进程 fork + 新 context）。

### 9.16.4 已被今天数据回答的一个缓解假设
参考配置本来就是 **2048 token 的分块预填充**（96K 请求要切成 48 块），
**而中毒照样发生** ⇒ 与 Astra 的保留意见一致：**分块延迟/摊薄并不能阻止它**，
真正相关的是"**累积映射工作集**"是否越过阈值，而不是单次突发的大小。
⇒ 可用的缓解方向应是「让总工作集不过阈值」（例如把超大上下文请求隔离到另一个进程/worker），
而不是"把请求切小"。

### 9.16.5 下一轮（一次重启，一次 96K 触发，全部一次性做完）
在 QSA op 探针里新增三组 arm，按 Astra 规范实现：
`T4`（VMM 4 格交叉，64–256 MiB 对象，低/高 MLP 各一组）、
`EC`（`empty_cache` 前后各跑一次 `FSC/STR/OE` 的恢复测试）、
`G2`（granularity 记录 + alias 多 VA span 粒度曲线），
保留同刻 `G4/BW/KRD` 稠密对照与 `OE` 生产对照，并把 CTA 时间分位数一并落盘。
这样一次运行就能把"VA / 物理 / context 级 / 分配池可恢复"四类判读一次性收紧。

## §9.17 决定性结果：载体是 **context/进程作用域的翻译-重放状态**（Astra #7 判读表命中"全格皆慢"）

### 9.17.1 为什么放弃裸 VMM 路线（两次把引擎打死，记录以免重犯）
在**引擎正在重放 CUDA 图的过程中**做驱动级 VMM 操作（`cuMemMap`/`cuMemUnmap`/DLPack 包装）会打死引擎：
- 第 1 版：边测边 remap ⇒ 第一个 T4 格就 `cudaErrorIllegalAddress`（**在任何 `empty_cache` 之前**）；
- 第 2 版：一次性建好 4 个映射、不再 remap ⇒ 仍 `cudaErrorAssert`；
- 并且 **vLLM 引擎的 logger 会丢弃 INFO 级消息**（`_xlog.info` 全不见，WARNING 才出现），
  我第一次因此无法定位——**探针里所有步进日志必须用 `warning` 级**。
⇒ 结论：**不要在引擎的 forward 里做驱动级 VMM 调用**；改用纯 torch 分配器的等价实验（下表），
一次成功拿到全部判读数据。VMM 代码（`_Vmm`、`_vmm_battery`）保留在补丁里但**不再被调用**；
`_mem_battery` 由 `/tmp/qsa_mem_arm` 触发。

### 9.17.2 电池设计（纯 torch，安全）
- `V0..V7`：**8 个独立新分配**的 64 MiB 缓冲区（互不相同、且实测跨越 **48 GiB** VA 空间），
  各做 4096 行 × 2 KiB 的随机行 gather（= 引擎 `FSC/STR` 的同一形状）；
- `D0..D7`：**同一批缓冲区**上的稠密连续拷贝（对照）；
- `VB`：1 GiB 大跨度缓冲区上的同形 gather（判"VA 跨度/页覆盖"）；`EC`：`empty_cache` 后新分配再测（恢复判据）。
- 干净态参考（一次性容器，同一补丁）：`V*=0.174–0.176 ms`、`D*=0.092–0.093 ms`、`VB=0.186 ms`。

### 9.17.3 中毒态结果（引擎进程内，触发后立即测量）

| arm | 中毒 | 干净 | 比值 |
|---|---|---|---|
| `V0..V7` 散列 8 个不同 VA（跨 48 GiB） | **39.68 / 39.68 / 39.68 / 39.68 / 39.68 / 39.68 / 39.68 / 39.68 ms** | 0.176 | **225×** |
| `D0..D7` 稠密拷贝（**同一批缓冲区**） | **0.090–0.092 ms** | 0.092 | **1.0×** |
| `GG` 引擎池式 gather（自建缓冲） | 39.7 ms | 0.174 | 228× |

（`free=0MiB` ⇒ 这些缓冲区来自引擎分配器池内空闲块；VA 跨度实测 51.6 GiB。）

### 9.17.4 判读（Astra 判读表）
- 8 个 VA 各差 6 GiB 以上、横跨 48 GiB，**全部同等慢到小数点后 3 位** ⇒
  **不是"某个 VA 区坏了"，也不是"某块物理内存坏了"，不是页子集问题**；
- 同一批缓冲区**稠密读同一瞬间满速（1.0×）** ⇒ 也不是"这些页不适合访问"；
- 结合 §9.15.5（同容器、同瞬间、**新进程健康**）⇒ 状态是 **per-context / per-process 的**，
  而不是设备全局的；
- ⇒ 命中 Astra 判读表的 **「四格都慢 ⇒ context 级翻译/重放压力或 process-scoped dxgkrnl 状态」**，
  并且由于"任何新分配都同样慢、任何 VA 都同样慢"，可进一步收紧为：
  **一个 context 作用域的、按"散列短请求"计费的翻译/重放资源**（不是按页、不是按区、
  不是按分配），而稠密流式访问完全不触发它。
- 这与 §9.14 的"单次访问完成延迟只差 1.05–2.7×、无 >10 µs 事件"并不矛盾：
  温度计只用了约 1 MiB 地址足迹、且 MLP 结构不同；本轮的 gauge 是**吞吐**，
  在**同样的缓冲区、同一瞬间**得到 225× vs 1.0× 的对照。

### 9.17.5 至此的完整排除清单（全部有实测）
硬件/DRAM 延迟、时钟与功耗刹车、ECC/坏行（Windows NVML）、PLE/SSD、Python 记账、
KV 池内容与放置、启动参数、数据值、网格/CTA 数量、内核身份、地址足迹 ≤3 GiB、
罕见的毫秒级单次访存延迟、WSL2 驱动错误路径（无 Xid/TDR/日志）、进程级"陈旧内存块"、
**VA 区、物理块、分配来源**。剩下唯一的载体类别就是上面那一条。

### 9.17.6 工程结论
- `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 不能去掉（仍值 30–35%）；
- 触发条件：**单个 ≥96K token 的请求**（分块 2048 也照样触发 ⇒ 不是突发，是**累积映射工作集**越阈）；
- 生效的"解药"只有**销毁那个 context**（重启引擎/容器，100% 有效）；
- 因此可操作的缓解是**不要让它发生**（把超大上下文请求隔离到另一个进程/实例），
  而不是"事后修复"或"把请求切小"。

## 9.18 最小复现器跨平台对比（"是不是 WSL 的锅"的第一次直接实验）

### 9.18.1 动机
用户要求：写一个最小复现器，看这个"分散小行访问崩塌"在 **Windows 原生**下会不会复现。
判据：若 Windows 原生也崩 ⇒ 问题在 Windows KMD/MCDM/dxgkrnl（两平台共有）；
若只有 WSL 崩 ⇒ WSL2 的 GPU-PV 客户层（uvm/dxgkrnl 的 GPU-PV 通道）是嫌疑。

### 9.18.2 复现器 #1（`~/vllm/minrepro.py`，纯 torch，同一文件两平台可跑）
- 臂：`scatter = dst.copy_(src[idx])`（16384 个随机 2 KiB 行，src = 32768×2048 uint8 = 64 MiB）、
  `dense = dst.copy_(src[:16384])`；载荷用不可压缩随机字节；
- 阶梯：基线 → 每 `--step` GiB 增常驻（ballast 分配 + `zero_()` 触碰 + `empty_cache` 保留）→ 到 `--gb`，
  每一步重测两臂 → 打印 ballast VA 范围 → 全部释放 + `empty_cache` 再测（恢复性）。
- 引擎停机腾显存后（真实 free = 65102 MiB），两平台各跑 `--gb 60 --step 6`：
  - **WSL2/Linux（同一容器镜像、同一驱动栈）**：scatter **0.160–0.187 ms（179–209 GB/s）每一步都不变**，
    dense 0.093–0.112 ms；释放后 0.166 ms。**全程无退化。**
  - **Windows 原生（torch 2.14.0+cu126，原生 KMD 路径）**：scatter **0.163–0.182 ms（184–206 GB/s）**，
    dense 0.092–0.112 ms；释放后 0.163 ms。**全程无退化。**
- ⇒ **"常驻映射工作集变大 + 分散小行访问" 本身不充分**；我这一版复现器抓不到引擎的触发条件。
  因此这次跨平台对比**无法判定平台**（两边都阴性）。

### 9.18.3 复现器 #2（`~/vllm/minrepro2.py`）——补上"引擎有而我上一版没有"的成分
新增两条阶梯（累积）：
- **pinned host memory** 0 → 4 GiB（`pin_memory=True` + `fill_()` 触碰 + 强制 device↔pinned 拷贝建立映射），
  覆盖 **GPU-PV / MCDM 的"主机内存被 GPU 映射"** 这条唯一必须打补丁才能用的路径；
- **分配 churn** 0 → 512 个新 2 MiB device 映射（对应 KV 块分配器的映射churn，而非几个大 ballast）。
结果（两平台同一文件、同参数）：
| | scatter | dense | pinned→dev |
|---|---|---|---|
| WSL2 基线 → pin 4 GiB / churn 512 | **0.100 ms 恒定** | 0.047–0.048 ms | ~2930 MiB/s 恒定 |
| Windows 原生基线 → pin 4 GiB | **0.100 ms 恒定** | 0.047–0.048 ms | ~3000 MiB/s 恒定 |
- ⇒ pinned host 映射与分配 churn **也都不是充分触发条件**；
- 附带事实：**两平台这些形状的性能几乎逐位相同** ⇒ 被测路径上没有可见的"WSL 税"，
  MCDM 原生能力与 GPU-PV 相当。

### 9.18.4 附带发现（重要，与后续判定有关）
1. `cuMemGetInfo` / `torch.cuda.mem_get_info()` 在这套平台上**谎报可用显存约 60 GiB**：
   引擎占用 61.5 GiB（Windows nvidia-smi: used 61657 / free 3446 MiB）时，
   **同容器内的新进程**仍报 `free 62.5 GiB`；**Windows 原生**也报 `free 62.58 GiB`。
   物理上只剩 3.4 GiB 时，申请 2 GiB 仍然成功。
   ⇒ MCDM 的显存会计是**按上下文**的（不认识别的 context），而 vLLM 正是用这个 API 决定 KV 池大小。
2. Windows 侧 `host_free` 只有 8.0 GiB（WSL VM 拿走了 64 GB），但不影响结论。
3. Windows 原生 torch 2.14.0+cu126 在 WSL 引擎占着 61.6 GiB 时仍能创建 context 并跑小内核 ⇒
   多 context 并存没问题。

### 9.18.5 判读（严格版）
- 否定的是："常驻大小 + gather 形状"或"pinned host 映射"或"分配 churn"**单独**足以造成崩塌；
- **没有**否定"context 作用域的翻译/重放资源"模型（新进程本来就免疫，结构上看不见）；
- 触发条件的缺失成分仍未找到 ⇒ 平台判定要等"能复现的最小实验"。

### 9.18.6 服务状态（本轮结束时）
- 引擎已恢复：容器 `a34efb2b7400`，`health=200`，新 2048 token 预填 **0.855 s（2396 tok/s）**，
  `num_requests_running=0`。
- 已向 Astra 发起第 9 次咨询（本节的负结果 + MCDM 事实 + `cuMemGetInfo` 谎报），
  请求：(a) MCDM vs WDDM 是否改变结论；(b) "缺失成分"候选排序及两平台可跑的判定性最小实验；
  (c) 在 vLLM 无法原生跑 Windows 的前提下，区分 WSL/GPU-PV 层 vs Windows KMD/MCDM 层 vs 引擎代码的最省钱方案；
  (d) Windows 侧只读可观测量（dxgkrnl 计数器 / ETW / GPUView / WMI）。

## 9.19 根因（终）：不是"分散访存崩塌"，是 **caching allocator 被推到 free=0 的碎片状态**

### 9.19.1 转折点：发现"临时张量"这个混淆因素
我此前所有"慢"的臂都是 `dst.copy_(src[idx])` —— advanced indexing 会先**分配一个临时输出张量**；
所有"快"的臂（切片/视图、Triton 内核、`out=` 形式、GEMM、温度计）**都不分配**。
而 `timed()` 是在**空队列**上用 CUDA event 夹逼的（`ev0.record(s)` → 内核 → `ev1.record(s)`），
所以**分配路径上的任何阻塞都会以"GPU 空等"的形式计入事件时间**。

### 9.19.2 判定电池（`_alloc_battery`，`/tmp/qsa_alloc_arm` + `/tmp/qsa_probe_arm`，默认配置，仪器版）
同一瞬间、同一批 buffer、同一数据：

| 臂 | 干净态 | **中毒态（首次探测）** | `empty_cache()` 之后 |
|---|---|---|---|
| T1 `copy_(src[idx])`（**要分配**） | 0.175 ms | **39.688 / 39.730 / 39.696 / 39.684 ms** | **0.173 ms**（此后一直正常） |
| T2 `index_select(src,0,idx,out=dst)`（**不分配**） | 0.090 ms | **0.089 ms** | 0.089 ms |
| T5 dense 视图（不分配） | 0.089 | 0.088 | 0.088 |
| T3 裸 `empty(32MiB)` host 耗时 | 0.005–0.01 | 0.005–0.012 | 0.005–0.012 |
| T1 的 **host 墙钟** | 0.06–0.08 | **0.28–0.37 ms（不是 39 ms）** | — |

内存记账：干净态 `alloc=57.17 GiB reserved=59.3 GiB free≈2.5 GiB`；
**中毒态 `alloc=57.17 reserved=63.16 GiB free=0MiB`**（`num_alloc_retries=0`、`segments 569→608`）。

### 9.19.3 因果确认（同容器、活体）
在该"中毒"容器里，分配电池的 `empty_cache()` 之后，**普通 2048 预填从 2.641 s 回到 0.725 s**
（随后连续三次都 0.72–0.73 s）⇒ **`torch.cuda.empty_cache()` 就地治愈，不需要销毁 context**。

### 9.19.4 机制
1. ≥96K 预填的**瞬时工作集**（chunk 激活/索引/workspace）把 reserved 从 59.3 GiB 顶到 **63.16 GiB**，
   设备 `free=0MiB`；
2. 缓存里的空闲块虽然总量有 ~6 GiB，但**已被切碎**，凑不出所需的连续块 ⇒ 每次要新分配的算子走到
   `cudaMalloc`，而此时设备已满 ⇒ 驱动侧回收/超额分配（落到系统内存/PCIe 路径），单次 **~40 ms**；
3. 于是**每一个需要新分配的算子**都付这 40 ms ⇒ 预填（每步都要临时激活）从 0.65 s 变成 2.1–2.6 s，
   而**不分配的算子**（dense 拷贝、Triton 内核、graph replay 里的固定地址）毫发无损。

### 9.19.5 这一下把之前的全部阴性/怪现象一次解释完
- "新进程免疫"：新进程显存空 ⇒ 分配永远能从缓存/设备内存满足；
- "所有 VA / 物理块 / 分配来源一样慢"：与地址完全无关，只与**要不要分配**有关；
- "每访问延迟正常（max 5.5 µs）"：温度计是 Triton 内核，**不分配**；
- "dense 快、gather 慢"：dense 用视图（不分配），gather 用 advanced indexing（要分配）；
- "grid/CTA 数量无关、数据无关、发射参数逐字节相同"：与这些无关；
- "随使用变差"：缓存碎片进一步恶化；
- "重启才恢复"（原来的结论）：其实是**初始化重新分配**；现在知道 `empty_cache()` 就够了；
- 无 Xid/TDR/驱动错误：分配器碎片不是错误路径。

### 9.19.6 解药（软件层，全部便宜）
1. 让 reserved 不要顶到物理上限：**降低 `--gpu-memory-utilization`**（0.96 → 0.94/0.93）或显式给
   `--kv-cache-memory-bytes`（Astra #1 的 7.5 GiB 建议正好在这个方向上）；
2. 改善复用、避免凑不出连续块：`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:<N>`；
   **`expandable_segments:True` 在这个引擎上启动即失败**（EngineCore 初始化错误，已实测）；
3. 必要时在长请求后**主动 `torch.cuda.empty_cache()`**（实测就地治愈）；
4. Astra #2/#5 的其余分配器后端只在上面都无效时再试。

### 9.19.7 待办
- [ ] A/B `QWEN_GPU_MEMORY=0.94`（进行中）；
- [ ] A/B `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`；
- [ ] 把"≥96K 后空闲为 0"这个真实机制从 OPS 全篇的旧结论里改过来（§9.14–§9.18 中"载体"表述作废，
      但所有**测量数据**仍然有效）。

### 9.19.8 修复验证（已实测）
| 配置 | 干净 2048 | 触发 96K | 触发后 2048 | 判定 |
|---|---|---|---|---|
| 默认 `QWEN_GPU_MEMORY=0.96` | 0.655–0.676 s | 51.7–58.1 s（1782–1900 tok/s） | **2.02–2.64 s（记录到 3.11×）** | 中毒 |
| `QWEN_GRAPH_MODE=NONE`（0.96） | 0.668–0.676 s | 51.7 s | 1.01 s（1.51×），**哨兵 V0=39.7 ms ⇒ 仍然中毒** | 只是暴露变浅 |
| `QWEN_SSD_PREFETCH=0`（0.96） | 0.671 s | 58.1 s | 2.09 s（3.11×） | 中毒 |
| **`QWEN_GPU_MEMORY=0.94`** | 0.648 s | **37.4 s（2630 tok/s，快 40%）** | **0.655 s（1.01×）** | **无中毒** ✅ |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | — | — | — | **启动失败**（EngineCore 初始化错误） |

结论：**把 reserved 留出余量（0.96 → 0.94）即可根除**；顺带让 ≥96K 预填快 40%。
运行中的容器 `f6c46ab97232` 就是 0.94 配置。

### 9.19.9 治愈线程（alloc healer）—— 完整上下文的验证
引擎进程内后台线程：当 `free < 384 MiB` 且 `cached > 256 MiB` 且距上次 ≥1 s 时调用 `torch.cuda.empty_cache()`
（实测就地治愈过：中毒引擎 2048 预填 2.641 s → 0.725 s）。只在该病态区间触发，健康时不动作。

| 配置 | 131072 预填 | 196608 预填 | 262143 预填 | 后置 2048 探针 |
|---|---|---|---|---|
| 0.94，**无** healer | 126.7 s（1035 tok/s） | — | — | 1.57 s / 2.2 s / 2.2–2.4 s ❌ |
| **0.94 + healer** | 51.9 s（2526 tok/s） | 78.8 s（2496 tok/s） | **105.5 s（2485 tok/s）** | **0.664 / 0.684 / 0.693 s（1.05–1.10×）** ✅ |
| 0.96 + healer | （本次验证中） | | | |

`QXHEAL` 全程触发 6 次（`free_before=54MiB cached_before=6283MiB` ⇒ 每次归还 ~6 GiB）。
关键：**预填吞吐在 131K/196K/262K 上恒定（2526/2496/2485 tok/s）⇒ 上下文长度不再造成退化。**

### 9.19.10 `empty_cache()` 的代价（为什么这个用法是安全的）
- 作用：把**未被使用的缓存块**通过 `cudaFree` 归还驱动；不触碰在用张量（权重/KV 池/激活/graph 池中在用的块），不搬运数据。
- 代价：①同步设备（毫秒级抖动）；②打破缓存复用⇒之后新分配要付 cudaMalloc（毫秒级）；③取分配器锁；④在 WDDM/MCDM 上释放+重分配偶发几十毫秒。
- 因此触发条件被收得很窄：只在"每次普通分配已经要付 ~40 ms"的病态区间动作 ⇒ 用几毫秒换掉持续的 40 ms 停滞。
- 更保守的替代（未采用）：把 `empty_cache()` 放在调度安全点（引擎 step 边界/空闲）而非后台线程，可完全避免与运行中内核抢锁。

## 9.20 用户桌面卡死事件（17:51 / 18:05）与配置修改（只改配置、不重启）

### 9.20.1 用户报告
物理显示器（Radeon RX550，3840×2160）+ 本机鼠标；**不是** GameViewer 远程（下午之前才是）。卡了"好一会儿"，**鼠标和窗口都不动，但应用窗口没有被关掉**。用户自己在 17:51 那次把卡住的 `GPU-Monitor.exe` 手动退出。

### 9.20.2 实机证据（Windows 事件日志 / 性能计数器，全部为只读采集）
- **`LiveKernelEvent 141`（GPU 看门狗活转储，`C:\Windows\LiveKernelReports\WATCHDOG\WATCHDOG-*.dmp`）今天 6 次**：13:15 / 13:44 / **14:08** / **14:16** / **14:24** / **16:18**，
  与我当天日志 mtime 一一对应：`run_vmm1.log`(14:07)、`run_vmm2.log`(14:15)、`run_mem1.log`(14:22)、`pabl_pn_graphnone.log`(16:18)+`minrepro4`(16:19)。
- **`nvlddmkm` id=153 `Error occurred on GPUID: b00`（Level=2）4 次**：14:16:37、14:24:03、15:32:30、16:18:37；其中 3 次出现在看门狗转储前 9–14 秒。15:32 对应当天 `minrepro3`（VMM map/unmap churn 4096/16384）。
- **17:51:24** 生成 `Kernel_0_0_0…` WER 内核活转储报告（含 `*.tmpatk.kdmp`）+ 17:51:25–26 共 60 条 WER 1001 `AppTermFailureEvent` + 用户手动退出 `GPU-Monitor.exe`（`Application Hang 1002`，17:52:26 自动重启）。
- **显示驱动全程无错**：`Display`/`amdkmdag`/`amdwddmg` 今天 0 条事件（只有 09-11 的信息级 4107），无 TDR 4101/4103 ⇒ **不是显示驱动崩溃**。
- **主机内存长期见底**：`vmmemWSL` WS 63.7 GB、**峰值 72.6 GB**；物理 88 GB ⇒ Windows 可用仅 **7.1 GB**；页面文件在用 C: 2.7 GB + D: 9.1 GB，**D: 峰值 39 GB**；WSL 侧 62.8 GB 中 **50 GB 是 page cache**，而 `.wslconfig` 未开 `autoMemoryReclaim`（VM 不回吐内存给主机）。
- **时间对齐**：17:51:25 落在 0.94+治愈线程阶梯容器 `246c602ed54a`（17:42:37 启动，日志 17:52:27 结束）的 **262143 长预填（105 s）**内；18:05:44 是我那个 2 秒的 `empty_cache` 计时容器——**在显存仅剩约 5 GiB 时又创建了第二个 CUDA context**（见 9.19.7 的纪律条目，属我的操作错误）。
- **既存不稳定史（与我无关的部分）**：09-26 有整串 WATCHDOG 转储（1053/1119/1122/…/2338）+ 一次蓝屏 `0xEF`（`Minidump\092626-22078-01.dmp`）；09-18/19 有 WHEA `124` 事件。

### 9.20.3 结论（当前最佳解释，未 100% 证实）
不是显卡崩溃，而是**主机层面的停顿**：显示卡无任何驱动错误；`LiveKernelEvent 141` 与 `GPUID: b00` 错误都指向 170HX 那条 NVIDIA 链路，且**全部与我的 VMM/内存电池实验同时**；17:51 那次无 141 转储，只有"内核活转储 + 应用被卡住" ⇒ 更像**主机停顿**（活转储抓取本身即会按住整机数秒）。放大因素是**主机可用内存只剩 7 GB + 页面文件活跃 + VM 峰值 72.6 GB**，此时驱动回收/驱逐（长预填把显存压到 free=0 时最重）很容易把整机拖住数秒。
待确认项：141 报告的"责任方"驱动名需要管理员读取 `C:\ProgramData\Microsoft\Windows\WER\ReportQueue\Kernel_141*\Report.wer`（当前 PowerShell 非管理员，读不到正文）。

### 9.20.4 本次修改（**只改配置，未重启任何东西**）
1. `C:\Users\hong\.wslconfig`：`memory=64GB` → **`memory=48GB`**，新增
   ```
   [experimental]
   autoMemoryReclaim=gradual
   ```
   备份：`C:\Users\hong\.wslconfig.bak-20260929`、`/tmp/wslconfig.bak-20260929`。**生效时机：下一次 WSL 关闭/启动时**（未执行 `wsl --shutdown`）。
2. `/home/hong/vllm/run_container.sh`：默认 `QWEN_GPU_MEMORY` **0.96 → 0.94**（第 50 行默认值、第 262 行 `-e`），保留 `QSA_ALLOC_HEAL=1` + `QSA_OPS_PATCH=/home/hong/vllm/qsa_ops_heal.py`；`bash -n` 通过，基础挂载 `-v "${MODEL_DIR}:/model:ro"` 未动。**生效时机：下一次 `run_container.sh` 启动容器时**。
   依据：0.94+治愈线程已实测 131072/196608/262143 全通过（探针 0.664/0.684/0.693 s，2485–2526 tok/s 恒定），显存不再被压到 free=0。
3. 当前仍在运行的容器 `36f5d1102f02` **仍是 0.96**（未重启），服务 `health=200`、`running=0`。

### 9.20.5 自我纪律（立即生效）
- 永不再跑 `_vmm_battery` / `_mem_battery`（今天 6 次 GPU 看门狗与之同行）。
- 永不在引擎占用显存时创建第二个 CUDA context（18:05 那个容器即此类错误）。
- 不再连跑 262K 阶梯；重实验前先问用户"现在能否用机器"。
- 不再使用 `nvidia-smi` 轮询（历史上失控循环 3h41m 即此类）。

## 9.21 主机内存"自己降下来"的真相 + 引擎 CUDA 崩溃与桌面冻结的因果链（2026-09-29）

### 9.21.1 为什么主机内存自己降了（`autoMemoryReclaim` 并未生效）
- **证据 1（我的配置确实没生效）**：客户机 `MemTotal` 仍是 **62.80 GiB**（若 `memory=48GB` 生效应为 ~46–47 GiB）。WSL 版本 2.7.11.0，内核 6.18.33.2，VM 连续运行 22h+ 未重启。
- **证据 2（是"还给 Windows"，不是客户机内部记账）**：客户机 `Cached` 50 GiB → 1.21 GiB 的同时，主机 `vmmemWSL` WS 63.7 GB → 14.3 GB，一一对应 ⇒ 页确实回到了 Windows。
- **证据 3（走的是 WSL 自己的回收路径，不是 Hyper-V 动态内存平衡器）**：客户机有 `hv_balloon` 模块；主机 `Get-Counter -ListSet 'Hyper-V Dynamic Memory VM'` 计数器存在但 **没有任何实例**（错误："指定的实例不存在"）。
- **证据 4（丢的是干净文件缓存，不是工作集）**：`pgsteal_direct` 仅 320 604 页、`SwapFree` 15.6/16 GiB 未动、`AnonPages` 前后都是 10.4 GiB。
- **结论**：**WSL 2.7.x 默认就开启了内存回收**（Microsoft `wsl-config` 文档：`autoMemoryReclaim` 默认 `dropCache` = 立即回收缓存），所以它检测到主机缺内存/缓存闲置时自己就把缓存丢掉并归还。我加的 `[experimental] autoMemoryReclaim=gradual` 因此**很可能没有作用**（另有资料称 2.7.3+ 已移除该设置项，本机是 2.7.11 ⇒ 该行可能被忽略；无副作用，但可删）。`memory=48GB` 仍然有意义：它把上限从 64→48 GB，**不让 page cache 再顶到 50 GiB**，从源头避免主机先掉到 7 GB 可用。

### 9.21.2 真正的元凶：引擎的 CUDA 崩溃 → 主机 GPU 看门狗活转储
主机目录 `C:\Users\hong\AppData\Local\Temp\wsl-crashes\`（文件名尾号 = 信号：6=SIGABRT，11=SIGSEGV）：

| 时刻 | 大小 | 信号/pid | 转储里挖出的原因 | 主机同时记录 |
|---|---|---|---|---|
| 09-29 11:49:05 | 4927 MB | ABRT / 95 | （未命中错误串） | — |
| **13:15:33** | 25.8 MB | ABRT / 95 | **`CUDA error: an illegal memory access was encountered`** | **GPU 看门狗活转储** |
| **14:16:42** | 53.1 MB | ABRT / 95 | **`CUDA error: device-side assert triggered`** | **看门狗活转储 + `nvlddmkm` GPUID b00** |
| **14:24:04** | 31.4 MB | ABRT / 95 | （错误串未命中） | **看门狗活转储 + GPUID b00** |
| **16:18:37** | 47.1 MB | ABRT / 96 | **`(EngineCore pid=96) ERROR RuntimeError: Triton Error [CUDA]: device-side assert triggered`** | **看门狗活转储 + GPUID b00** |
| 17:55:18 | 1058 MB | SEGV / 1 | （被强杀） | — |

- 13:15/14:16/14:24 与我当天的 `run_vmm1`(14:07)、`run_vmm2`(14:15)、`run_mem1`(14:22) 对应；16:18 对应 GRAPH_MODE=NONE 消融与 `minrepro4`。
- **因果链（已由证据闭合）**：我的引擎内探针/电池 → 引擎以 **CUDA 非法访存 / 设备侧断言** 崩溃（SIGABRT）→ 主机 NVIDIA 驱动记 `Error occurred on GPUID: b00` → `dxgkrnl` 抓 **WATCHDOG 活内核转储**（抓取期间整机冻结数秒~数十秒）→ **用户桌面与鼠标冻结**。此外 WSL 每次还把 **1–5 GB 的崩溃转储写到主机磁盘**，在主机仅剩 ~7 GB 可用内存时是二次重击。
- 17:55:18 那份 1.06 GB 转储是 `docker rm -f` **强杀运行中的引擎**造成的（在活动内核之上拆除 60 GiB CUDA 上下文）。

### 9.21.3 本次修改与纪律
- `run_container.sh`：容器停机改为**优雅停机**——先 `docker stop -t 30 "$NAME"`，再 `docker rm -f`（第 240 行附近）；帮助文本同步更新；`bash -n` 通过。
- 纪律追加：**引擎内任何会执行 CUDA 内核的探针/电池（`_vmm_battery`/`_mem_battery`/`_alloc_battery`/QXPROBE/QXGATE 路径）永久退役**；不再有任何"让引擎崩一次看看"的实验；强杀引擎一律避免。
- 待办：①确认 `autoMemoryReclaim` 在 2.7.11 是否被忽略（若是则从 `.wslconfig` 删掉该行）；②可选清理 `wsl-crashes` 里 8.3 GB 的转储（均是我的引擎崩溃产生的，C: 余 66.6 GB）。

## 9.22 去 Docker：原生 WSL 部署已就绪（2026-09-29，尚未切换）

### 9.22.1 用户要求
`1` `.wslconfig` 删掉 `autoMemoryReclaim` 行（**已做**，保留 `memory=48GB`）；`2` 删除 `wsl-crashes` 转储（**已做**，释放 8.2 GB，C: 67G→75G）；`3` 不再用 Docker，把 vLLM 相关代码/配置拷到项目目录下，直接在 WSL 里跑。

### 9.22.2 已完成的准备工作（对运行中的 Docker 服务零影响）
- 拷贝目标：`/home/hong/code/qwen3.8-flash-next-cmp170hx/vllm-native/`（8.1 GB）
  - `opt/vllm/{.venv,src,optimization}`（8.0 GB，含 torch 1.1G / nvidia 轮子 3.0G / tokenspeed_triton 294M / flashinfer 113M / src 下 18 个已编译 `.so`）
  - `opt/{entrypoint.sh,download-model.sh,qwen38-ssd}`；`uv-python/cpython-3.12.14-linux-x86_64-gnu`（110 MB，venv 的基解释器）
  - **跳过** `/opt/nvidia`（1.3 GB，只有 nsight-compute，本机 profiling 全部不可用，用不到）
- 拷贝正确性：`find -type f | wc -l` 镜像 68860 / 3732 与拷贝**完全一致**。
- 路径改写：61 个 `venv/bin/*` 的 shebang、`pyvenv.cfg` 的 `home`、`__editable__*.pth`（`/opt/vllm/src` → 本地）、`direct_url.json`、`uv-python/cpython-3.12-linux-x86_64-gnu` 符号链接、`venv/bin/python` 链接。
- `qsa.py` 拷进来的就是**治愈版**（含 `_start_alloc_heal`）；原版另存 `qsa.py.stock`（来自 `/tmp/qsa_ops.py`）。
- 启动脚本：`vllm-native/bin/run_native.sh start|stop|status|foreground`（默认 0.94 + 治愈线程 + `VLLM_WSL2_ENABLE_PIN_MEMORY=1` + 端口 9393；复用 `/home/hong/vllm/triton_cache`，避免重新编译；stop 为 SIGTERM 优雅停机）。
- **校验（不触碰 GPU：`CUDA_VISIBLE_DEVICES=` 置空）**：`bin/run_native.sh` 语法 OK；`vllm serve --help` 正常；`ple_ssd_io.so` 可 dlopen；导入清单与镜像**逐项 diff 为空**（torch 2.13.0+cu130 / triton 3.7.1 / vllm 0.29.1rc1.dev402+ga5a30471f.ple1 / flashinfer 0.6.18 / tokenspeed_triton 3.8.10 / `_C_stable_libtorch` / `_custom_ops` / `_flashmla_C` / `fs_io_C` / qsa 全部 OK；`flash_attn`、`vllm._C`、`ray` 在镜像里同样不存在 ⇒ 非漏拷）。

### 9.22.3 切换（**已于 2026-09-29 执行完成，见 §9.23**）
- ~~真正的切换（`docker stop -t 60 hong-pc` → `run_native.sh start` → 验证 health 200 + 一次全新 2048 预填 + 解码抽测）~~ → 已执行，见 §9.23。
- 回退一条命令：`run_native.sh stop && cd /home/hong/vllm && ./run_container.sh`（Docker 路径完全未改）。
- 说明：Docker 自身启动只占几秒，5–7 分钟是加载 143 GB 权重；原生的实际收益是不依赖 docker 守护进程/镜像层、日志与进程直连、`/dev/shm` 从容器 64 MB 变为 VM 的、便于用 nsys/gdb/py-spy 调试。
- 注意：切换后若执行 `wsl --shutdown`（例如 `memory=48GB` 生效时）或重启电脑，原生引擎需要重新 `start`；用户已明确**不需要**开机自启。

## 9.23 Docker → 原生切换完成与验收（2026-09-29）

用户指令："切换吧，端口模型保持不变，不需要开机自启"。

### 9.23.1 切换过程
1. 切入前确认空闲：`vllm:num_requests_running=0`（排除自己的残留请求干扰）。
2. `docker stop -t 60 hong-pc` → 约 2 s 退出，`Exited (0)`；`restartpolicy=no` ⇒ **不会被 docker 自启抢端口/抢卡**；端口 9393 释放。
3. `vllm-native/bin/run_native.sh start`（pid 1621365，`local:1621365` 独立会话 = 已脱离我的 shell）。
4. **第一次启动失败**：`PermissionError: [Errno 13]` 写 `/home/hong/vllm/triton_cache/<hash>/`（该缓存由容器以 root 写入，原生进程是 `hong`）⇒ EngineCore init failed。
   - 修复：把热的 135 MB 缓存复制成 `vllm-native/triton_cache`（362 目录，hong 可写，哈希不变 ⇒ 不触发重编译）；`run_native.sh` 改为 `TRITON_CACHE="${TRITON_CACHE_DIR:-$ROOT/triton_cache}"` 并加**可写性预检**（失败时提示 `cp -r /home/hong/vllm/triton_cache/. $ROOT/triton_cache/`）；`bash -n` 通过。
   - 老缓存 `/home/hong/vllm/triton_cache` **保留**（仍属 root，供将来 Docker 回退使用）。
5. 第二次启动成功：`/health` 200 用时 **321 s**（加载 143 GB 权重），`QXHEAL alloc-heal thread started`，FlashAttention 2 + ple_ssd 路径，11 条 CUDA graph 日志。

### 9.23.2 验收数据（全部为原生引擎实测，2026-09-29）
| 项目 | 原生 | Docker 基线 | 判定 |
|---|---|---|---|
| 2048 全新预填（热） | **0.635 / 0.680 / 0.687 s**（3224 tok/s） | 0.632–0.689 s | ✅ 一致 |
| 8192 全新预填 | **2.024 / 2.089 / 2.092 s**（4047 tok/s） | ~2.1–2.3 s | ✅ 略优 |
| 131072 长上下文 | **54.629 s**（2399 tok/s） | 53.541 s（2450） | ✅ 一致 |
| 131072 之后全新 2048（中毒检查） | **0.680 / 0.687 s** | 需治愈 | ✅ 无中毒 |
| 纯解码（`ignore_eos`，两点差减） | **106.3 tok/s** | ~111 tok/s | ✅ 差 4% |
| 治愈线程 | `QXHEAL fired #1: free_before=332MiB cached_before=5974MiB` | 8 次 | ✅ 工作 |
| `/v1/models` | `Qwen3.8-Flash-Next`，`max_model_len=262144` | 同 | ✅ |
| 显存 | 63761 / 65536 MiB，GPU1 = 0 MiB | 同 | ✅ |
- 精确解码测法：`max_tokens=64/128/256` + `ignore_eos:true`，`(256-64)/(T256-T64)` ⇒ 剔除 TTFT。

### 9.23.3 ⚠️ 唯一的行为变化：Windows 侧必须用 `127.0.0.1`，不能用 `localhost`
- 实测（Windows PowerShell → WSL）：`http://127.0.0.1:9393/health` **200** ✅、`http://127.0.0.1:8000/v1/models` **200** ✅、`http://localhost:9393/health` **FAIL（超时）** ❌、`http://[::1]:9393/health` **FAIL** ❌。
- 原因（已用两台临时 HTTP 服务对照实验定位，非猜测）：
  - 已在 WSL 里绑 `0.0.0.0:9999`（**双栈 `::`**，含 IPv4-mapped）→ Windows `localhost:9999` 仍 **FAIL**，`127.0.0.1:9999` OK。
  - 已在 WSL 里绑 `::1:9998`（纯 IPv6 回环）→ Windows 两种写法**都 FAIL**。
  - ⇒ `networkingMode=mirrored` 下 **Windows 的 IPv6 回环（::1）根本不会转发进 WSL**，只转发 IPv4 回环 127.0.0.1。Windows 的 `localhost` 优先解析成 `::1` ⇒ 必然超时。（Docker 时代 `localhost` 能用，是因为端口由 Windows 侧的 docker-proxy 监听，那是 Windows 自己的 IPv4+IPv6 双栈监听，与 WSL 无关。）
  - ⇒ **改绑定地址无济于事**（对照实验已证），故**无需**为它重启引擎。
- 想让 `localhost` 继续可用只有三条路，**都需要用户权限/重启，本次均未做**：① 客户端 URL 改 `127.0.0.1`（零成本，推荐）；② 管理员权限给 `C:\Windows\System32\drivers\etc\hosts` 加 `127.0.0.1 localhost`；③ `.wslconfig` 改回 `networkingMode=NAT`（需 `wsl --shutdown`，NAT 下 WSL 会在 Windows 侧建立双栈监听）。
  - 另：Windows 侧想自建 9393 的 IPv6 中继**不可能**——实测 `bind(('::1',9393))` 返回 `WSAEACCES 10013`，该端口已被 mirrored 模式下的 WSL 监听占用保留。

### 9.23.4 切换后的运维约定
- 状态/启停：`cd /home/hong/code/qwen3.8-flash-next-cmp170hx/vllm-native && bin/run_native.sh status|start|stop`（stop 为 SIGTERM 优雅停机，60 s 宽限）。
- 日志：`vllm-native/logs/server.log`（无需 `docker logs`）；pidfile `logs/server.pid`。
- `127.0.0.1:8000 → 9393` 转发器（pid 49733，`~/dlvenv/bin/python ~/vllm/forward8000.py`）仍在跑，Windows 侧 `127.0.0.1:8000` 同样 200。
- Docker 回退路径未做任何破坏性改动（`run_container.sh` 完好，镜像仍在本地）；`docker daemon` 仍在跑但容器 `Exited (0)` 且策略 `no`，不会抢卡。
- `wsl --shutdown` / 重启电脑后需手动 `run_native.sh start`（用户不需要开机自启）。
- 遗留待办：`.wslconfig memory=48GB` 仍**未生效**（guest 仍 62 GiB，需一次 `wsl --shutdown`，等用户发话）；`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`、`--kv-cache-dtype fp8_e4m3` 等可选优化未做。

## 9.24 宿主内存被抽干 + "解码变慢"的真相（2026-09-29）

### 9.24.1 现象 A：切换后宿主机可用内存只剩 6.4 GB
- 实测：`vmmemWSL` 工作集 **62.2 GB**（宿主共 87.9 GB）、Windows 可用 **6.4 GB**、页面文件 C: 2.7 GB + D: 8.8 GB（峰值 38 GB）。
- 原因（已量化，非猜测）：原生启动要读 **143 GB 权重**，内核把它全当干净页缓存留下 ⇒ guest `Cached = 50595 MiB`（≈50 GB），balloon 把这 50 GB 全算进 vmmemWSL 工作集。
  **注意：这不是"原生 vs Docker"的差别，而是"每次加载模型都会发生"**，只是切换时正好又发生一次。
- 处理（**不重启、不动引擎**）：WSL 启动器允许免密指定 root，所以不需要 sudo 密码：
  `wsl.exe -d Ubuntu -u root -- sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
  效果：`vmmemWSL 62.2 → 19.8 GB`、`Windows 可用 6.4 → 48.7 GB`、guest `free 1342 → 51019 MB`。
  事后验证引擎无碍：health 200、全新 2048 预填 **0.681 s**（基线 0.635–0.687）、解码指标不变、
  **解码期间磁盘读 = 0 MiB**（PLE 走 `O_DIRECT` + 自己的 512 MiB 行缓存，不经内核页缓存 ⇒ 丢缓存伤不到它）。
- 封装：`vllm-native/bin/drop_host_cache.sh`（幂等，可反复跑；`bash -n` 通过）。

### 9.24.2 现象 B：用户觉得 decode 变慢 —— **步时没变，是 MTP 接受率在变**
测量脚本 `~/vllm/dec_bench.py`（流式 + `stream_options.include_usage` + 引擎自己的 spec-decode 计数器算接受率 + `/proc/diskstats`）：

| 运行 | MTP 接受率 | 纯解码 tok/s | 步间 p50 | p95 | 解码期磁盘读 |
|---|---:|---:|---:|---:|---:|
| warm1（64 tok，短窗） | 80.0 % | 96.6（首步 75 ms 离群拉低小样本） | 16.1 ms | 20.3 ms | 0 MiB |
| run2（256 tok） | 67.3 % | **104.2** | 15.9 ms | 17.2 ms | 0 MiB |
| run3（256 tok） | 68.4 % | **106.9** | 15.7 ms | 16.9 ms | 0 MiB |
| run4（短 prompt） | 60.4 % | 100.9 | 15.9 ms | 17.1 ms | 0 MiB |

- **结论**：步时（step latency）**没有变慢**——原生 15.7–16.1 ms/step，对照 OPS.md 里 Docker 时代的记录
  `decode 单流 59.8~66.1 chunk/s` = **15.1–16.7 ms/step**，完全重叠。
- `tok/s = (1 + 接受率) / 步时`：接受率 78 % ⇒ 1.78/0.0159 = **112.3 tok/s**（就是当年那个 112.27 基线）；
  接受率 68 % ⇒ 1.68/0.0159 = **105.7**；接受率 60 % ⇒ 101。**观察到的差异 100% 由接受率解释**，
  而接受率由**内容**决定（中文散文/结构化文本高，长推理与代码通常低）。
- 排除项：解码期磁盘读 0 MiB（PLE 不在关键路径）、p95 步间 17 ms 无抖动、治愈线程未触发、
  env 与启动参数和容器**逐项 diff**只差 CUDA 工具链变量和我的 shell 变量（`VLLM_WSL2_ENABLE_PIN_MEMORY=1`、
  `OMP_NUM_THREADS=1`、全部 serve 参数一致）⇒ 无配置回归。
- 可选提速（需重启，≈5.4 分钟，**待用户同意**）：`--speculative-config num_speculative_tokens: 1 → 2`
  （MTP 深度 2，理论上限 3 tok/step；实际收益取决于第 2 位接受率，通常 ~1.9–2.0 tok/step ≈ 120–126 tok/s）。

## 9.25 MTP 投机解码深度 1 vs 2：A/B 实测（2026-09-29）

用户要求："提到 2 再测试一下"。结论：**MTP=2 比 1 快 13.5%，已设为默认**。

### 9.25.1 ⚠️ 先说测量协议的教训（差点得出相反结论）
第一轮 MTP=2 测出 94.9 / 96.2 / 105.9 tok/s（步时 p50 21.8~22.6 ms），比 MTP=1 的 104~107 还慢，
看起来像是"提高深度反而变慢"。**这是假象**，因为那批数据是在两个坏条件下取的：
1. 引擎**刚启动**（只跑了几个 prefill，没有解码预热）；
2. 宿主机处于**内存高压**（`Windows 可用 6.9 GB`、`vmmemWSL 57.8 GB`、页面文件狂换）。
   等 balloon 缩回（可用 51.4 GB）后同配置复测，步时立刻从 22.2 ms 掉到 17.87 ms、120.1 tok/s。

⇒ **以后任何解码对比，必须满足**：① 预热 ≥2500 个解码 token；② `vmmemWSL < 32 GB`（跑 `bin/drop_host_cache.sh` 后等 20~60 s）。
否则同一配置可以测出 ±25% 的假差异。这也解释了 9.24 里"步时 15.7~16.1 ms"为什么当时是稳的——那批数据虽然宿主也在高压，
但引擎已跑过大量请求（相当于预热充分）。

### 9.25.2 公平 A/B（两边都充分预热 + 宿主低压，同一批 prompt）
| 指标 | MTP=1 | MTP=2 | 变化 |
|---|---:|---:|---:|
| 每步出 token 数（含接受的 draft） | 1.70 | **2.20** | +29 % |
| 步间延迟 p50 | 15.2~15.7 ms | 17.2~17.6 ms | +2.0 ms/步 |
| 步间 p95 | 16.1~16.8 ms | 18.4~21.1 ms | — |
| 稳态纯解码吞吐 | 106.3 / 108.2 / 110.0 / 111.1 / 113.8 / 116.0 | **118.4 / 121.2 / 122.9 / 125.6 / 126.6 / 126.7 / 130.1 / 130.7 / 130.9** | **≈111 → ≈126 tok/s（+13.5 %）** |
| 引擎计数器接受率（accepted/drafted） | 63.5~77.1 % | 51.6~64.9 %（每步 draft 2 个，故平均每位更低） | — |
- 结论：多出的那次 draft 前向只花 ≈2 ms，却多换 ≈0.5 token/步，净收益为正。
- 顺带确认：**预填不受影响**——MTP=2 下 2048 = 0.641 s、8192 = 2.114/2.123/2.147 s（与 MTP=1 的 0.635/2.024~2.092 同档）。
- 旁注：两种深度都偶发**单步 ~1.3 s 长尾**（各自 4 次运行里出现 1 次，`max=1331 ms`），
  不影响平均吞吐；同期 `QXHEAL fired = 0`，与内存治愈线程无关。
- 未测 MTP=3：边际收益递减（第 3 位接受率通常 ~35~45%），而每次切换都要 4~5 分钟重启，等用户发话再说。
- 回退：`QWEN_MTP=1 bin/start.sh`（或把 `vllm-native/bin/run_native.sh` 里 `QWEN_MTP` 默认值改回 1）。

## 9.26 目录迁移：~/vllm → 项目内 ops/ 与 bin/（2026-09-29）

用户要求："之前 docker 时创建的 ~/vllm 现在还有用吗？有用的搬到项目目录下；再写服务启动/停止/日志脚本"。

### 9.26.1 迁移结果（`~/vllm` 已删除，867 MB 全部归位）
| 原位置 | 新位置 | 说明 |
|---|---|---|
| `~/vllm/{warmup,dec_bench,prefill_ab,prefill_step,gridsweep,footprint}.py` | `ops/bench/` | 在用 |
| `~/vllm/{forward8000,fdl,dl_blobs,dl_model,verify_ckpt,verify_sha256,resume.sh,vllm_logs.sh,analyze_trace,stage_parse,progress.sh,fdl_supervisor.sh}` | `ops/tools/` | 备用 |
| `~/vllm/{minrepro*,*_instr.py,ablation.sh,allocprobe.sh,ladder.sh,probe_ablation.sh,run_minrepro4.sh,*_probe.py,gathertest.py,graph_cost.py,cuda_overhead.py,ple_latency.py,enable_profiler_entrypoint.sh}` | `ops/diagnostics/` | **已退役**，只读留档 |
| `~/vllm/measurements/`、`~/vllm/*.json` | `ops/measurements/` | 证据 |
| `~/vllm/prof/` | `ops/prof/` | kineto trace |
| `~/vllm/*.log` | `ops/logs/` | 历史日志 |
| `~/vllm/{run_container.sh,qsa_ops_heal.py,triton_cache/,download-complete}` | `ops/legacy-docker/` | Docker 回退用 |
| `~/vllm/src/`（**含 13 个未提交补丁**） | `ops/src-upstream/src/` | 719 MB，勿删 |
| `~/vllm/{.venv,.vscode,__pycache__}` | `ops/legacy-docker/junk/` | 100 KB 空 venv 等 |
- 引用修正：`run_container.sh`（挂载/补丁/OPS 路径）、`run_native.sh`（triton 缓存提示）、`ops/diagnostics/*.sh`、`ops/bench/prefill_*.py`。
  全部 `bash -n` 通过；`run_container.sh` 引用的 5 个宿主路径（`qsa_ops_heal.py`、`triton_cache`、`ops/prof`、`enable_profiler_entrypoint.sh`、模型目录）逐个验证存在。
- `.gitignore` 增补，防止 9 GB 运行资产入库：`vllm-native/`、`ops/src-upstream/`、`ops/legacy-docker/triton_cache/`、`ops/legacy-docker/junk/`、`ops/{logs,measurements,prof}/`。
  **坑**：`.gitignore` 不支持行内注释（`vllm-native/   # 说明` 会被当成一个带空格和 `#` 的模式而失效），注释必须单独一行。
- 结果：git 未跟踪的只剩 58 个脚本/文档文件（`bin/*.sh`、`ops/bench/*`、`ops/diagnostics/*`、`ops/tools/*`、`ops/OPS.md`、`ops/README.md`）。

### 9.26.2 新增服务脚本（项目根 `bin/`）
| 脚本 | 作用 |
|---|---|
| `start.sh` | 启动 → 轮询 `/health` 直到 200（默认上限 600 s，实时显示已用时间）→ 打印接口/进程/显存 → 自动跑 `drop_host_cache.sh` → 拉起并托管 :8000 转发器（`--no-forward` 可关，`--keep-cache` 不回收内存，`--foreground` 调试） |
| `stop.sh` | 先查 `num_requests_running`（`--check` 只报告），有人用会要求确认；再 SIGTERM→最多 60 s→SIGKILL；验证端口释放、显存归零、转发器停止 |
| `status.sh` | 健康码、pid/运行时长/RSS、运行中/排队请求、MTP 接受率、显存、宿主内存（Windows 可用 + vmmemWSL）+ guest 内存、日志大小；`--watch N` 刷新、`--short` 单行 |
| `logs.sh` | `-f` 跟踪、`-n N`、`-e` 错误筛选、`--heal` 内存治愈线程、`--startup` 启动关键行、`--slow`、`--list` 列归档与可读大小、`--path`、`--clean` |
| `bench.sh` | 一键体检：新鲜预填（2048/8192，`--full` 加 131072 + 中毒后置检查）+ 解码 tok/s + 接受率 |
| `drop_host_cache.sh` | 用 `wsl.exe -u root`（免密、不需要 sudo）丢 WSL 干净页缓存，把内存还给 Windows |

### 9.26.3 启动器（`vllm-native/bin/run_native.sh`）改进
- `QWEN_MTP` 默认 **2**（附 A/B 数据注释）。
- 新增 `restart` 子命令；`start` 时日志轮转（>100 MB 换名，只留最近 5 份）。
- 仍在 Triton 缓存不可写时报错并给出正确修复路径（指向 `ops/legacy-docker/triton_cache`）。

### 9.26.4 交付状态（本次结束时）
`health=200`、模型 `Qwen3.8-Flash-Next`、`max_model_len=262144`、`QWEN_MTP=2`、
预填 2048 = 0.641 s / 8192 = 2.11 s、解码 ≈126 tok/s、显存 60401/65536 MiB、
宿主 `free=53.2 GB / vmmemWSL=13.4 GB`、`:8000` 转发器 200、Docker 容器仍为 `Exited (0)`（回退路径完好）。

## 9.27 性能数据正式归档 + 目录合并（2026-09-29）

用户要求：① 把性能数据记录保存成文件；② 弄清 `vllm-native/` 与 `ops/` 的区别并合并重复部分。

### 9.27.1 性能数据现在存在哪
| 文件 | 内容 |
|---|---|
| `docs/RESULTS-WSL2.md` | **主记录**：本机（WSL2 原生，无 Docker）全部性能数据 + 与参考机 pass2 的逐项对比 + 测量注意事项 |
| `results/pass3-wsl2-native/final-wsl2-bench.json` | 项目自带 `benchmarks/bench_server.py` 产出，128 token、3 轮中位数、并发 1/4/8 |
| `results/pass3-wsl2-native/final-wsl2-long-bench.json` | 512 token 扫描，同上 |
| `results/pass3-wsl2-native/final-wsl2-prefill.json` | 新鲜 seed 的 512/2K/8K/32K 预填，3 轮 |
| `results/pass3-wsl2-native/final-wsl2-smoke.json` | 6 项服务体检（含韩语、工具调用、16.8K 检索） |
| `results/pass3-wsl2-native/manifest.json` | 硬件/版本/补丁/serve 参数/env 快照 |
| `ops/measurements/perf-history.csv` | **自动追加**：`bin/bench.sh` 每次运行写一行（时间、tag、MTP 深度、预填 2048/8192/131072、解码中位数、步时 p50、接受率、宿主内存、原始日志名）+ `bench-*.log` 原始输出 |

正式数字（3 轮中位数，MTP=2，与参考机 pass2 同协议）：
- 单请求解码 **131 tok/s**（128 token）/ **132 tok/s**（512 token）；参考机 110.99 / 116.88 ⇒ **+18 % / +13 %**
- 并发 4 聚合 **305 tok/s**（参考 277.22）；并发 8 聚合 304（参考 428.97，**低**：本机 `max-num-seqs=4`，参考机用 16 序列，8 个请求被分两波，TTFT 也因此被抬到 2.9 s）
- 预填中位数 512→0.266 s、2048→**0.693 s**、8192→**2.184 s**、32768→**10.404 s**；参考机 0.317 / 1.069 / 3.108 / 15.372 ⇒ **快 1.2~1.5 倍**（本机 NVMe + 88 GB 内存 vs 参考机 SATA + 15 GB 内存，PLE SSD 都是瓶颈）
- 长上下文：131072 预填 54.63 s（原生）/ 53.54 s（Docker，+治愈线程）；196608 → 80.06 s；262143 → 104.70 s，后置全新 2048 = 0.68 s ✔（QXHEAL 触发 8 次）

### 9.27.2 测量方法上的两个坑（已写进文档，避免以后误判）
1. **`bench_prefill.py` 的 seed 固定**（`seed + length*7 + repeat`）⇒ 第二次用同样 seed 再跑，
   prompt 与上次完全相同 → **命中前缀缓存**：8192 从 2.184 s 掉到 0.418 s，32768 从 10.404 s 掉到 0.677 s。
   实测对照（全新 seed vs 紧接再跑）：8192 = 2.261/2.091 s → 0.529/0.418 s；32768 = 10.379/10.661 s → 0.677/0.677 s。
   **结论：每次都换 `--seed`。**（我第一版 pass3 数据因此被污染，已用 `--seed 20260929` 重跑覆盖。）
2. 解码必须"预热 ≥2500 token + `vmmemWSL < 32 GB`"，否则步时虚高 25~40 %（详见 9.25.1）。

### 9.27.3 目录合并：`ops/src-upstream` 已删除（-719 MB）
- 用户问："`vllm-native` 和 `ops` 都有 vllm，能不能合起来？"
- 事实：`vllm-native/opt/vllm/src` 是**运行中的那份** vLLM（958 MB，含编译好的 `.so`）；
  `ops/src-upstream/src` 是**上游 git 克隆**（719 MB，其中 .git 占大头），当年用来开发/对照补丁。
- 验证等价（关键证据）：`git -C ops/src-upstream/src apply --check --reverse patches/qwen38-ple-ssd.patch`
  **通过** ⇒ 仓库自带的 `patches/qwen38-ple-ssd.patch` 完整覆盖克隆里那 13 个未提交改动
  （11 个修改 files + 2 个新增 `ple_ssd.py`/`ple_ssd_io.c`；文件名集合逐一比对，只差那 2 个 untracked）。
  克隆 HEAD = `a5a30471ff`，无本地提交，因此删除后**没有丢失任何独有内容**。
- 补丁能力现在由两份补丁 + 文档保证：`patches/qwen38-ple-ssd.patch`（上游项目补丁，1345 行）
  与新增的 `ops/patches/qsa-alloc-heal.patch`（**本机独有的内存治愈补丁，之前只存在于运行的 qsa.py 里**，
  从 `qsa.py.stock` → `qsa.py` 生成，65 行，已用 `patch --dry-run -R -p1` 反向校验）。
  重建步骤写在 `ops/README.md`。
- 结果：项目里只剩**一份** vLLM 源码；`ops/` 从 867 MB 降到 149 MB；磁盘余量 144 GB。

### 9.27.4 新增/修改
- `bin/bench.sh` 重写：解码前自动预热 2560 token、宿主高压时告警、结果自动写入
  `ops/measurements/perf-history.csv` + `bench-<时间戳>.log`，支持 `--tag/--full/--prefill/--decode/--no-save`。
  修过一个解析 bug：`TPOT=nan ⇒ 0.0 tok/s` 被当成解码值，导致中位数算成 55.35（现只匹配 `纯解码=… ⇒ X tok/s`）。
- `ops/bench/dec_bench.py`：TPOT 为 `nan` 时不再打印那一段（该构建的 `time_per_output_tokens` 直方图为空）。
- `ops/README.md` 重写：新增"vllm-native 与 ops 的区别"对照表、数据位置索引、两个测量坑、重建补丁步骤。

## 9.28 会诊机制固化：`AGENTS.md` + `ops/tools/consult.sh` + 9 次会诊归档（2026-09-29）

### 9.28.1 起因
用户问："还记得之前是怎么和 gpt-6 讨论的吗？把使用场景和方法写进 `AGENTS.md`。"
⇒ 目标：把"卡住时找 gpt-6 会诊"这条有效经验，从**散落在 OPS.md 里的引用**变成**可复制的机制**。

### 9.28.2 证据核实（不是靠回忆，逐条查过）
- **9 次会诊的原件全部还在 `/tmp`**：`astra_brief{,2..9}.md`、`astra_reply{,2..9}.md`、`astra_err{,2..9}.log`。
  时间戳显示每次耗时 **15~25 分钟**（#9：15:08 发 → 15:25 回）。
- **err 日志内容**（105 B，无害）：`[pi-web-access] Dynamic tool activation requires Pi 0.86.0 or newer; web tools remain eagerly available.`
- **brief 的结构**（以 #9 为模板）：`标题` → `Established engine-level facts` → `NEW result #1/#2/#3`
  → `What I want from you (be concrete, quantitative, rank by expected value per unit cost)`，正文用 (a)(b)(c) 编号提问。
- **CLI 环境**：`pi 0.85.1`（`~/.local/share/pi-node/node-v22.23.1-linux-x64/bin/pi`）；
  `pi --list-models` 确认 `lmstudio → gpt-6-astra`（250K 上下文）、
  `openrouter → openai/gpt-6-astra(-pro)`（1.1M）为备选。

### 9.28.3 今天重跑验证（关键，别再猜）
| 试验 | 结果 |
|---|---|
| `--thinking minimal` | ❌ **HTTP 400**：`level "minimal" not supported, valid levels: low, medium, high, xhigh, max` |
| `--thinking low` | ✅ 返回 `PONG` ⇒ **原命令路径（`lmstudio/gpt-6-astra`）今天仍然有效**，只是档位不能写 `minimal`/`off` |
| `ops/tools/consult.sh` 冒烟（thinking=low，小 brief） | ✅ `rc=0`，回复落盘 `CONSULT-OK`，21 秒完成；期间修掉 2 个真 bug（见 9.28.5） |

### 9.28.4 交付物
1. **`AGENTS.md`（仓库根）** —— agent 工作手册，四节：
   - §1 项目速览（模型/引擎/目录/接口）
   - **§2 与 gpt-6 讨论：什么时候该问（触发条件）+ 怎么问（命令、时间、读回复）+ brief 骨架
     + 必须写进 system prompt 的约束 + 拿到回复后怎么回填 OPS.md**
   - §3 铁律 10 条（不验证不算完成、别用 `localhost`、解码测量协议、换 seed、中毒检查、
     探针永久退役、回收宿主页缓存、优雅停、重启要问、`~/vllm` 已废弃）
   - §4 常见任务速查 + 基线数字（判异常用）
2. **`ops/tools/consult.sh`** —— 一键会诊：`--list`（历史 9 次一览）/ `--check <编号>`（是否写完）
   / `--wait`（前台等 + 进度 + 结果汇报）；自动附加只读约束、自动归档 brief、
   产物 `reply-N.md`/`.err`/`.rc`（`.rc` 内容 = 真实退出码，作为完成标记）。
3. **`ops/consultations/`** —— 把 `/tmp` 里的 9 份 brief + 9 份 reply + err **归档进项目**
   （`/tmp` 会被清理，这是唯一一份留档）：`brief-01..09.md`、`reply-01..09.md`、`reply-01..09.err`。
4. `ops/README.md` 增加"和 gpt-6 会诊"小节与 `consultations/` 索引；`AGENTS.md` 指向该节。

### 9.28.5 冒烟测试抓到的两个真 bug（写脚本的价值）
1. `local n="$1" rc="$OUT/reply-$n.rc" rp="$OUT/reply-$n.md"` —— bash 的 `local/declare/export`
   会**先把整条命令的所有赋值词展开**再赋值，所以同一句里 `$n` 取不到刚赋的值；
   在 `set -u` 下直接 `n: unbound variable` 退出。⇒ 改成一行一个 `local`。
2. 回复文件在 `pi` 写出之前不存在，进度行 `wc -c < "$REPLY"` 会报
   `No such file or directory`（重定向错误由 shell 报，`2>/dev/null` 拦不住）。
   ⇒ 启动前 `: > "$REPLY"` 预创建。

### 9.28.6 经验（以后照做）
- 会诊的 brief 要**把已确立的事实带数字复述一遍**，否则顾问会重复讨论历史；
  并明确要求"最便宜的决定性实验 + 判读矩阵"，否则回复容易变成泛泛建议。
- 回复必须落到 `OPS.md`：写成"**它判定 X → 我实测 Y → 因此 Z 成立/被否证**"（§9.11–§9.18 就是范例）。
  与实测冲突时以实测为准，并把冲突记下来（历史上出现过，§9.12/§9.14）。
- 会诊期间**不要**顺手重启引擎：约束写进 system prompt 是为了防止顾问建议"重启试试"被照做。

## 9.29 删除 Docker 预构建镜像（释放 25 GB，2026-09-29）

### 9.29.1 用户要求
"可以把 docker 镜像删掉了吧" —— 删 `18gogogo/170hx1-qwen38nextf:sm80`。

### 9.29.2 删之前做的三项验证（不验证不删）
1. **原生路径是否依赖镜像** —— 用 `grep -rn "18gogogo|/var/lib/docker|/workspace" vllm-native/bin/*.sh`（无命中）
   + `find vllm-native -xtype l`（**0 个指向外部的软链接**）+ 启动器路径全在 `vllm-native/` 与
   `/home/hong/models/...` 之内 ⇒ **自包含，删镜像不影响服务**。
2. **镜像还能不能拉回来** —— `docker manifest inspect 18gogogo/170hx1-qwen38nextf:sm80` 退出码 0、
   manifest 4166 B、**22 层** ⇒ Docker Hub 仍匿名可用，re-pull 有保障。
3. **容器可写层有没有独有产物** —— `docker diff hong-pc`：4276 新增 / 668 修改，除去 WSL 自动挂载的
   `/usr/lib/wsl/drivers`（NVIDIA DLL）后只剩 `/root/.humming/cache`（编译缓存）与
   `/root/.config/vllm/usage_stats.json` ⇒ 无独有产物（`qsa_ops_heal.py` 早已归档到 `ops/legacy-docker/`）。

### 9.29.3 做了什么
- 先写下**还原凭据** `ops/legacy-docker/IMAGE-PROVENANCE.md`：Image ID
  `sha256:9d8f3babf23360d46009f9c17a6747ce4c953e67169ffc4f9071a7ae5c338bd3`、RepoDigest、
  创建时间 `2026-09-26T05:06:30Z`、22 层 digest 全列表、`Size=8.61 GiB`，
  以及"按 digest 拉 + 比对 Image ID"的校验方法（Docker 内容寻址 ⇒ ID 相同即内容相同）。
- `docker rm -f hong-pc`（`run_container.sh` 本来就会 rm -f 重建同名容器，故无影响）
- `docker rmi 18gogogo/170hx1-qwen38nextf:sm80` ⇒ `Untagged` + `Deleted: sha256:9d8f3bab…`
- **磁盘：144 GB → 169 GB 可用（812 → 788 GB 已用，实收 25 GB）**。
  ⚠️ **此处原写法（"discard 会立即归还宿主"）已于当晚被实测否证**：`/` 虽带 `discard`，但 WSL 的
  `D:\wsl\ext4.vhdx` 文件长度不缩、**Windows 的 D: 可用空间一分未增**（21:24 与 22:0x 两次测得均为 317 GiB，
  期间 WSL 内部又释放了 42.7 GB）⇒ **释放的空间只对 WSL 内部有效，要还给 Windows 必须压缩 vhdx**
  （`wsl --shutdown` + `diskpart` → `select vdisk file="D:\wsl\ext4.vhdx"` → `attach vdisk readonly` → `compact vdisk`）。
  潜在可回收量 ≈ vhdx 长度 842 GiB − ext4 实际已用 740 GiB ≈ **100 GiB**。
- 服务未受影响：删后 `bin/status.sh --short` = `health=200 pid=1631819 up=33:38 running=0.0`。

### 9.29.4 连带更新（避免照旧文档踩空）
- `AGENTS.md`：§1 部署说明改为"镜像已删、回退需先 pull"；§3 新增铁律 11（Docker 回退 = 先
  `docker pull <digest>` 再 `run_container.sh`，并指向凭据文件）。
- `ops/README.md`：回退章节改为"先 pull（含 digest 命令）+ 校验 + 再运行"，并说明容器已删、
  回退仍需保留的宿主文件（`run_container.sh`、`qsa_ops_heal.py`、`triton_cache/` 135 MB）。

### 9.29.5 没有动的东西（属于用户其它项目，需用户自己决定）
| 对象 | 大小 | 说明 |
|---|---|---|
| `ghcr.io/syv-ai/hyperqwen:latest` | 14.6 GB | 另一个项目的镜像 |
| `ghcr.io/syv-ai/qwen38-27b-rtx3090:latest` | 14.7 GB | 同上 |
| `qwen38-27b-3090:latest` | 14.7 GB | 同上 |
| 卷 `qwen38-27b-rtx3090_qwen-cache` | 17.09 GB | 2026-08-22 创建（另一个模型的缓存） |
| Build cache | 40.34 GB（其中可回收 12.95 GB） | `docker builder prune` 可回收 |
| `alpine:3.20` + 3 个 `nvidia/cuda:*` base | ~1.7 GB | 重建镜像时可能要用，建议留着 |

## 9.30 2026-09-29 20:13–21:19 主机连续三次非正常关机 = NVMe 掉盘（2026-09-29）

### 9.30.1 事件
用户在 21:23 报告"电脑突然断电重启"，并补充"再次启动识别不到固态硬盘，过了一会再次重启才恢复"。
这次崩溃把正在运行的引擎（pid 1631819）连同 WSL 一起干掉，我的一条命令（对 docker 卷做递归 `du/find`）
也中断（返回 "No result provided"）。

### 9.30.2 时间线（全部来自 Windows 事件日志，非推测）
| 时刻 | 事件 |
|---|---|
| 20:03:40 | 我删掉 170HX 镜像（-25 GB）——`ops/legacy-docker/IMAGE-PROVENANCE.md` 落盘时刻 |
| 20:08:38 | 我删掉两个 27b 镜像（-29 GB）——`ops/DELETED-DOCKER-IMAGES.md` 落盘时刻 |
| 20:08–20:11 | 我对 `/var/lib/docker/volumes/…/_data` 做递归 `du`/`find`（元数据密集 I/O，全在 D: 上的 vhdx 内） |
| **20:11:45** | **`stornvme` 129：Reset to device, `\Device\RaidPort2`, was issued**（NVMe 控制器复位）← 第一条硬件异常 |
| **20:12:51** | **`stornvme` 129：第二次复位** |
| **20:13:46** | **非正常关机 #1**：BSOD `0x7A KERNEL_DATA_INPAGE_ERROR`(Arg2=`0xC000000E` STATUS_NO_SUCH_DEVICE) |
| 20:14:45 | 自动重启；20:14:50 `volmgr` 162 dump 写入成功；20:15:35 WER 1001 记录 `C:\WINDOWS\Minidump\092926-51437-01.dmp` |
| **20:18:37** | **非正常关机 #2**（Event 41 `BugcheckCode=0` ⇒ 连蓝屏都出不来） |
| **20:20:43** | **非正常关机 #3**（同样 `BugcheckCode=0`） |
| 20:20:43→21:19:16 | **约 58 分钟起不来**（用户：BIOS/Windows 认不到那块 SSD） |
| 21:19:16 | Windows 启动（`LastBootUpTime`）；21:21:10 WSL 起来；21:24 现场检查 |

### 9.30.3 机制（结论）
1. **盘从总线消失**：`stornvme` 129 = 存储栈命令超时后复位控制器；随后 Windows 在需要换入页面时
   拿到 **STATUS_NO_SUCH_DEVICE** ⇒ 蓝屏 0x7A。**WSL 的 vhdx 就在这块盘上**（`D:\wsl\ext4.vhdx`，
   841.4 GiB），所以 `KERNEL_DATA_INPAGE_ERROR` 会直接命中正在运行的 WSL/引擎。
2. **掉盘到 BIOS 认不到**：控制器固件锁死，需要断电周期才恢复 —— 与用户的观察完全一致。
3. 后两次关机 `BugcheckCode=0`（没有转储）也符合"盘已经不在了"：写不出转储。
4. **没有 WHEA-Logger / PCIe AER 事件**（近 3 天 0 条）⇒ 不是 CPU/平台级机器检查异常，
   问题定位在**该 NVMe 设备/链路**层面。
5. 盘身份：**KIOXIA EXCERIA G2 1.86 TB NVMe**（DRAM-less/HMB），固件 **ECFA17.3 = 官方最新**
   （官方发布页确认最新就是 17.3，只覆盖 17.0/17.1 升级；该版本修改的正是 WCTEMP 温度阈值）
   ⇒ **"升级固件"这条路已经用尽**，剩下的可解释因素是 PCIe 电源管理（ASPM/APST）、温度、M.2 接触/插槽、盘本体老化。
6. 本机还有更早的不稳定背景：近 7 天 **384 条 `nvlddmkm` 事件**、5 次 bugcheck
   （09-26 ×3：`0x3B`/`0x3B`/`0xEF`；09-28 `0x3B`；09-29 `0x7A`）——今晚这次是**存储侧**的新问题。

### 9.30.4 与我（agent）操作的因果关系：只能说时间相关，不能说是根因
- 我的两个删除动作（合计释放 54 GB）与随后的递归扫描都发生在 **20:08–20:11**，
  紧接着 **20:11:45 第一次控制器复位**（间隔约 3 分钟）。
- 但用户态删文件**不会**让设备从总线消失（更不会让 BIOS 认不到盘）；这类症状属盘固件/供电/链路层面。
  已知诱因中，"**大块 TRIM 突发 + 高随机 I/O**"确实能让边际控制器锁死 —— 所以我的操作是
  **可能的时间相关触发载荷，不是根因**。
- 教训（写进 `AGENTS.md`）：**引擎在跑时不做几十 GB 级删除/全盘扫描**；housekeeping 一律停机做。

### 9.30.5 崩溃后现场完整性（已逐项检查，无损）
| 检查 | 结果 |
|---|---|
| ext4 | `EXT4-fs (sdd): 1 orphan inode deleted` + `recovery complete`；`lost+found` **0 条目** ⇒ 无文件丢失 |
| 模型 | 13 个分片齐全（PLE 95.37 GiB + 10×4.66 GiB + 3.88 GiB + mtp 0.49 GiB），dmesg 无 I/O 错误 |
| 运行时 | `vllm-native/opt/vllm/.venv` 在；triton 热缓存 158 MB；`vllm-native` 外部软链接 **0** |
| 仓库 | `git status` 与崩溃前一致（只有脚本/文档改动） |
| docker | daemon 正常；三个大镜像删除**已生效**；`qwen38-27b-rtx3090_qwen-cache`（17 GB）**仍在**（尚未删） |
| 引擎 | 已随重启停止；**清理了僵尸 pid 文件**（`server.pid`=1631819 / `forward.pid`=1633061 均为死进程），
`bin/status.sh --short` 现在诚实显示 `health=000 pid=-`；9393/8000 无监听 |

### 9.30.6 建议（按性价比排序，未执行，待用户决定）
1. **管理员读 SMART**（普通权限拿不到，`Get-StorageReliabilityCounter` 报 "Access to a CIM resource was not available"）：
   `powershell -Command "Get-PhysicalDisk | Get-StorageReliabilityCounter | Format-List *"`
   ⇒ 看 `MediaErrors`/`Wear`/`Temperature` + NVMe Critical Warning，区分"盘要坏了"还是"健康但链路/供电不稳"。
2. **关 ASPM / 改 PCIe 电源管理**（NVMe 负载下掉盘的经典修复，需要一次重启）：
   BIOS 里把该 M.2 的 ASPM 设为 Disabled（或 L0s/L1 disabled）；Windows 电源计划 → PCI Express →
   **链接状态电源管理 = 关闭**；顺带关"硬盘空闲后关闭"。
3. **物理层**：重插 M.2、检查散热（EXCERIA G2 无 DRAM，对温度敏感；机箱里还有 CMP 170HX）。
4. **工作纪律**：引擎运行时不做 GB 级删除/全盘扫描；大清理先 `bin/stop.sh`。
5. **风险提示**：项目的全部运行时（841 GiB vhdx + 143 GB 模型 + docker）都在 D: 这块盘上，
   而 C:（INTEL SATA 477 GB）只剩 80 GiB ⇒ **没有就地迁移空间**；若 SMART 显示盘在劣化，
   应优先考虑把 vhdx/模型迁到大容量可靠盘（涉及采购，按约定不由我提议）。

### 9.30.7 SMART 数据（用户在管理员 PowerShell 里读到，2026-09-29 21:3x）
| 盘 | Temperature | TemperatureMax | Wear | 其它 |
|---|---|---|---|---|
| **1 = KIOXIA EXCERIA G2（D:，掉盘的那块）** | **53 °C** | **72 °C** | **0** | 无读写错误计数上报；链路 **Gen3 x4（current=max，未降速）**；PCI bus 1 |
| 0 = INTEL SSDSC2KW512G8（C:，SATA） | 0（不支持） | 0 | 23 | 通电 1855 h；LoadUnload 35885；ReadLatencyMax 1004 ms |

判读：
- **`Wear=0` ⇒ 不是写磨损/寿命问题**（这块盘几乎没写坏）；也没报读错误。
- **温度是唯一的异常项**：**待机 53 °C、历史峰值 72 °C**。作为对照，评测与用户报告显示
  EXCERIA G2（=RC20 后续型号）在**加了散热片**时满载仅 45 °C、不散热能到 60~65 °C+；
  厂商 2024-06 的 ECFA17.3 固件说明里改的正是 **WCTEMP（警告复合温度阈值）**，说明该型号对热阈值敏感。
- Windows 的 `MSFT_StorageReliabilityCounter` **不暴露** NVMe 的 Critical Warning / Media Errors /
  Unsafe Shutdown Count / Available Spare ⇒ 要看这些必须装 **KIOXIA SSD Utility** 或 **CrystalDiskInfo**（免费，非采购）。

### 9.30.8 拓扑与电源管理（都不是管理员就能查）
- KIOXIA 控制器：`PCI bus 1, device 0, f0`，**Gen3 x4**；GPU 在 `bus 11/12`（两张 CMP 170HX，Gen1 x8）
  ⇒ **与 GPU 不共享上游链路**，排除"GPU 侧扰动"。
- 活动电源计划 = 平衡；**AC 下 "PCI Express → 链接状态电源管理" 已经是 0（关闭）**
  ⇒ Windows 软件侧 ASPM 已经是最保守设置，`powercfg` 不是可用的杠杆；
  剩下的电源状态风险只可能在 **BIOS 的 ASPM/L1 子状态** 与 **盘固件自主的 APST**（操作系统看不到）。
- 近 2 天无 CPU 热保护/降频事件；近 3 天无 WHEA/PCIe AER 事件。

### 9.30.9 公开案例对照（同症状）：这是"NVMe 主控锁死"的经典形态
- r/pctroubleshooting："PC crashed and SSD disappear on BIOS until power cycle"
- LTT 论坛：随机冻结 → BSOD → 重启后 BIOS 里 SSD 消失 → **"只有彻底断电才回来"**
- Level1Techs：**"在 lspci/BIOS 里都看不到，直到完全断电冷启动"**
⇒ 与本机症状（掉盘 → BIOS 认不到 → 断电后恢复）逐条对应；属**盘/主控层面的锁死**，
不是文件系统或驱动配置问题。另有公开报道称 Windows 11 24H2（本机 build 26100.4061）改过
HMB（Host Memory Buffer）分配方式，在 HMB 类 SSD 上引发 `KERNEL_DATA_INPAGE_ERROR (0x7A)`
——**与本机 bugcheck 码完全一致，但该说法未被权威来源证实，仅列为待验证候选**。

### 9.30.10 复现/监控工具
`ops/tools/ssd_temp_log.ps1`（纯 ASCII，需管理员）：按秒记录 KIOXIA 的温度/峰值/磨损/延迟，
用来回答"我们的负载是否把这块盘推到 70 °C+"这个决定性问题。

---

## 9.31 脚本整理：端口/参数默认值收敛到唯一来源 + 顺手修掉 4 个真 bug（2026-10-01）

### 9.31.1 起因（用户提问链，都是真现象）
1. "日志能看出服务在哪个端口吗" → 看到 `Starting vLLM server on http://0.0.0.0:9393` 与 `'port': 9393`。
2. "我用 start.sh 默认启动的，怎么会被设置成 9393？" → 查证后：**9393 本来就是真默认**
   （`vllm-native/bin/run_native.sh:17` `PORT="${QWEN_PORT:-9393}"`），
   反而是 `bin/_common.sh:15` 写成了 `8001`。
3. "怎么会出现这种乱七八糟的情况，改个端口都不知道在哪里改" → 本次整理。

### 9.31.2 实测证据：端口错值的真实后果（不是洁癖）
```
$ bin/status.sh --short          # 整理前
health=000 pid=605789 running=- waiting=- kv=- acc=-      ← 探的是 :8001
$ curl :9393/health  → 200       ← 引擎其实健康
$ curl :8001/health  → 000
```
* `health_code()` 永远 000 ⇒ **所有依赖 health 的脚本误判"服务没起"**。
* 最危险的是 `bin/start.sh` 的守卫 `if [ "$(health_code)" = 200 ]`：服务在跑时再敲一次
  `./start.sh` 会去起**第二个引擎**抢显存（踩铁律 1）。
* `bin/stop.sh` 收尾的"端口已释放 ✔"看的也是 8001 ⇒ 假确认。
* 另：`bin/bench.sh:71` 硬编码 `127.0.0.1:9393`（明明已 source `_common.sh` 却不用 `$BASE_URL`）；
  `ops/tools/forward8000.py` 硬编码 9393/8000；`ops/tools/metrics_web.sh:23` 第三份默认值。

### 9.31.3 做了什么（唯一来源 + 分层）
* 新增 `config/engine.env`：**端口与全部引擎参数的唯一默认值来源**，写法统一
  `: "${QWEN_PORT:=9393}"`（环境变量优先）。被 `bin/_common.sh`、`run_native.sh`、
  `ops/tools/metrics_web.sh` source；`ops/bench/*.py` 经 `ops/bench/_cfg.py` 读它。
* `bin/_common.sh` 改成"只做路径派生 + 函数"，不再持有任何默认值；新增 `usage()`
  （把脚本头部注释块原样打印）与 `print_params()`。
* `--help` 全面改为 `help_exit "$0"`：**不再用 `sed -n '2,Np'` 数行号**（这是"加一行注释 help 就错位"的根源）。
* `run_native.sh` 新增 `print-cmd`（干跑）与 `params`；`stop` **默认不再 SIGKILL**（要强杀须显式 `QWEN_FORCE=1`）。
* 新增 `docs/SCRIPTS.md`：调用链、脚本清单、遗留件说明、新增脚本 4 条硬约定。

### 9.31.4 顺手修掉的 4 个真 bug
| 位置 | 原状 | 后果 |
|---|---|---|
| `bin/logs.sh --clean` | 注释写"归档并清空"，实现只有 `: > log` | 静默丢日志（见 9.31.5） |
| `run_native.sh` 缓存修复提示 | `$(cd "$ROOT/.." && pwd)` 路径算错 | 提示里给的是不存在的目录 |
| `run_native.sh stop` | 60 s 后自动 SIGKILL | 与铁律 8 冲突 |
| `ops/tools/forward8000.py` | 全程静默且端口写死 | `forward.log` 一直 0 字节，挂掉无任何线索 |

### 9.31.5 附带发现：对"被重定向打开的日志"做截断会留 NUL 空洞（决定性证据）
`bin/logs.sh --clean` 在引擎运行时执行后，`server.log` 变成 56385 字节、其中 **56304 字节是 NUL**：
```
$ grep -E '^flags:|^pos:' /proc/605789/fdinfo/1
flags:	0100001      ← O_WRONLY|O_CREAT，**没有 O_APPEND(0x2000)**
pos:	56385        ← 进程自己记着的写入偏移
```
机制：`setsid nohup vllm … > server.log` 是普通重定向（非 O_APPEND），进程持有偏移量；
截断后下一次写入落在旧偏移上 ⇒ 文件头出现等长空洞。（`run_native.sh` 原有的启动轮转用 `mv`，
因为发生在**新引擎启动前**、无人持有偏移，所以一直没暴露这个问题。）
**修法**：① 启动改为 `>>`（O_APPEND）⇒ 之后 `--clean` 可安全 copytruncate（logrotate 语义）；
② `--clean` 先读 `/proc/<pid>/fdinfo/1` 的 flags，没有 `0x2000` 就**只归档不截断**并说明原因；
③ 本次留下的空洞已在文件头写入 477 字节说明文字（只覆写空洞区间，偏移 56304 之后的真实日志未动，
总大小不变 56385）。

### 9.31.6 验证（不改语义的决定性检查：干跑 vs 线上进程逐项 diff）
```
$ diff <(tr '\0' '\n' < /proc/$(cat vllm-native/logs/server.pid)/cmdline | tail -n +2) \
       <(vllm-native/bin/run_native.sh print-cmd)
★ 完全一致（37 项参数一个不差）
```
其余：`bash -n` 全部通过；6 个 Python 文件 `py_compile` 通过；
`bin/status.sh --short` → `health=200 pid=605789 acc=63.2%`；
`QWEN_PORT=1234 bin/status.sh --short` → `health=000`（证明覆盖生效、全链同源）；
真实预填（新探针路径，不带任何环境变量）：`2048 tok best of 2 = 0.740 s`
（第 1 次 0.868 s 含 Triton 惰性编译，末次 0.740 s 落在基线 0.64~0.73 s 上沿，属正常）。
整个过程**没有重启引擎、没有跑 GPU 重负载**。

### 9.31.7 经验
* **默认值只允许有一份**：一旦同名变量在两处各写一个默认值，二者迟早会分叉，
  而且症状会以完全不相干的形式出现（这里是"status 说服务没起"）。
* **能用"干跑打印 + 与线上进程 diff"验证的改动，优先这么验**：比重启一次便宜 4~5 分钟，
  且结论是二值确定的（一致/不一致），不依赖人读日志。
* **凡是要截断在用的文件，先问它的 fd 是不是 O_APPEND**（`/proc/<pid>/fdinfo/N` 的 flags）。
  `mv` 与 `: >` 两种轮转写法对"谁持有 fd"的假设完全不同，混用就会留下空洞。

---

## 9.32 端口统一到 8000 + 删除转发层（2026-10-01）

### 9.32.1 用户要求
> "默认端口直接改成 8000，不需要再转发了"

原本是**双端口**结构：引擎听 9393，再用 `ops/tools/forward8000.py` 做 `:8000 → :9393`
纯字节转发（因为 Windows 侧部分客户端只认 8000）。既然引擎可以直接占 8000，
转发层就是纯多余的故障点（而且它的 `forward.log` 一直是 0 字节、挂了没有任何线索）。

### 9.32.2 做了什么
1. `config/engine.env`：`QWEN_PORT` 9393 → **8000**；删掉 `QWEN_FWD_PORT`。
   因为是唯一来源，`bin/*`、`run_native.sh`、`metrics_web.sh`、`ops/bench/*.py` 自动跟随。
2. **删除 `ops/tools/forward8000.py`**（含 `vllm-native/logs/forward.{pid,log}`，日志本来就是空的）。
3. 摘除转发相关的全部代码：`bin/_common.sh`（`FWD_*` 变量、`fwd_code/fwd_pid/fwd_alive`）、
   `bin/start.sh`（`--no-forward` 选项、拉起/探测转发器的整段）、`bin/stop.sh`（`--keep-forward`、
   停转发器整段）、`bin/status.sh`（"转发器 :$FWD_PORT" 一行）。
4. 文档同步：`AGENTS.md`（接口行 + 速查行）、`docs/SCRIPTS.md`（清单/遗留表/背景）、
   `ops/tools/metrics_web.sh` 头部、`ops/bench/_cfg.py` 注释。

原转发器源码（12 行，留档以便将来真要用时恢复）：
```python
import asyncio
async def pipe(r, w):
    try:
        while (d := await r.read(65536)):
            w.write(d); await w.drain()
    finally:
        w.close()
async def handle(cr, cw):
    try:
        rr, rw = await asyncio.open_connection("127.0.0.1", 9393)
    except Exception:
        cw.close(); return
    await asyncio.gather(pipe(cr, rw), pipe(rr, cw), return_exceptions=True)
async def main():
    s = await asyncio.start_server(handle, "127.0.0.1", 8000)
    async with s:
        await s.serve_forever()
asyncio.run(main())
```

### 9.32.3 影响面（改端口不是改一行就完事）
* **必须重启引擎才生效**：`--port` 是启动参数，已写进运行中进程的 cmdline。铁律 9 ⇒ 重启要用户点头。
* **在重启完成前，`bin/status.sh` 会显示 `health=000`**（探针已指向 8000，而引擎还在 9393）——
  这是预期的过渡态，不是故障。谁都不许在这个窗口里按健康检查下判断。
* **`:9494` 看板需重启一次**：正在跑的看板进程（pid 59248）启动参数里写死了
  `--upstream http://127.0.0.1:9393`，改端口后要 `ops/tools/metrics_web.sh stop && start`。
* **外部客户端**：凡是指向 9393 的调用方（脚本、浏览器书签、别的项目配置）都要改成 8000。
  已查本仓活跃路径无残留；本机 pi 的 provider 是 lmstudio，与本引擎无关。
* 顺带受益：`benchmarks/*.py` 的默认 URL 本来就是 8000，现在它们指向的就是引擎本身。

### 9.32.4 验证
* 改动后 `bash -n` 全部通过；`bin/_common.sh`/`start.sh`/`stop.sh`/`status.sh` 里
  已搜不到任何 `FWD|forward|转发` 残留（只剩 config 里一行删除说明）。
* `grep -rn "QWEN_PORT"` 全仓：默认值仅 `config/engine.env` 一处（+ `_cfg.py` 的等值兜底）。
* **已完成**（用户同意后于 2026-10-01 11:56–12:01 执行）：`bin/stop.sh && ./start.sh` ⇒ 详见 §9.32.6。

### 9.32.5 经验
* 端口这种"谁都可能引用"的常量，**先收敛到一个来源再谈改动**：9.31 收敛完之后，
  这次改端口 + 删转发层只碰了 1 个配置值 + 4 个脚本的删除行，没有再出现"漏改一处"。
* 删中间层要同时删掉**围绕它的运维手势**（选项、状态行、pid/log 文件），
  否则会留下"看起来还在、其实永远无响应"的僵尸检查项——那正是 9.31 里 8001/9393 的教训。
