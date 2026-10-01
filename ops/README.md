# ops/ —— 运行、测量与排障资产索引

本目录是原来 `~/vllm/` 的全部有用内容（2026-09-29 迁移，原目录已删除）。
**日常操作请用项目根目录的 `bin/`**；**引擎本体在 `vllm-native/`**。

## vllm-native/ 和 ops/ 的区别（常见疑问）

| | `vllm-native/` | `ops/` |
|---|---|---|
| 是什么 | **运行时**：真正跑起来的 vLLM（venv 7.1 GB + 已打补丁的 vLLM 源码 958 MB + 编译好的优化库 + Triton 热缓存 + 日志），从镜像 `18gogogo/170hx1-qwen38nextf:sm80` 拷出来 | **运维资产**：测量脚本、证据、补丁归档、历史日志、Docker 回退件、文档 |
| 删了会怎样 | 服务起不来 | 服务照跑，但丢了工具/证据 |
| 谁在跑 | `bin/start.sh` → `vllm-native/bin/run_native.sh` | 被 `bin/*.sh` 调用 |

**为什么以前"两处都有 vllm"**：`vllm-native/opt/vllm/src` 是**运行的那份**；
`ops/src-upstream/src` 曾是一份上游 vLLM 的 git 克隆（719 MB，用来开发补丁、看 diff）。
那份克隆里的 13 个改动 = 仓库里已有的 `patches/qwen38-ple-ssd.patch`（已用
`git apply --check --reverse` 验证完全等价），因此**克隆已删除**，现在项目里只有一份
vLLM 源码（运行中的那份）。要重建补丁能力：

```bash
git clone <upstream vLLM> && cd vllm && git checkout a5a30471ff   # 上游基线
patch -p1 < ../../patches/qwen38-ple-ssd.patch                    # 上游项目补丁（13 个文件）
patch -p1 < ../../ops/patches/qsa-alloc-heal.patch                # 本机内存治愈补丁（qsa.py）
```

## 日常操作（入口在 bin/）

```bash
cd /home/hong/code/qwen3.8-flash-next-cmp170hx
./start.sh          # = bin/start.sh 的根目录薄封装（启动 + 等就绪 250~320 s + 回收宿主内存 + 托管 :8000 转发器）
bin/status.sh       # 健康/请求/显存/宿主内存/MTP 接受率（--watch 5 / --short）
bin/logs.sh -f      # 日志（-e 错误 / --heal 内存治愈 / --startup 启动行 / --list 归档）
bin/bench.sh        # 体检，并自动追加到 ops/measurements/perf-history.csv
bin/stop.sh         # 优雅停止（--check 先问会不会打断用户请求）
bin/drop_host_cache.sh   # 把 WSL 页缓存还给 Windows（防整机发卡）
```

调参用环境变量：`QWEN_MTP`（默认 2 = 最快）、`QWEN_GPU_MEMORY`（0.94）、
`QWEN_CONTEXT`（262144）、`QWEN_SEQS`（4）、`QWEN_BATCH_TOKENS`（2048，**别改成 8192**）、
`QWEN_PORT`（9393）。例：`QWEN_MTP=1 bin/start.sh`。

## 数据与文档在哪里

| 位置 | 内容 |
|---|---|
| `docs/RESULTS-WSL2.md` | **本机（WSL2 原生）全部性能数据**：与参考机 pass2 的逐项对比、MTP 深度 A/B、预填/解码/长上下文、测量注意事项 |
| `results/pass3-wsl2-native/` | 正式原始数据（JSON，用项目自带 `benchmarks/` 产出）+ `manifest.json` |
| `ops/measurements/perf-history.csv` | `bin/bench.sh` 每次运行追加一行（时间、MTP、预填、解码、接受率、宿主内存、原始日志名） |
| `ops/OPS.md` | 全部排查过程与结论（1600+ 行），含被推翻的假设与证据 |
| `docs/RESULTS.md`、`docs/PERFORMANCE.md`、`results/pass2/` | 参考机（15 GB RAM / SATA / Docker）的原始数据，用于对比 |
| `ops/consultations/` | **和 gpt-6-astra 的 9 次会诊原件**（brief/reply/err）+ `ops/tools/consult.sh`（下次会诊用） |
| `AGENTS.md`（仓库根） | agent 工作手册：什么时候该会诊、命令怎么发、brief 怎么写，以及本项目的 10 条铁律 |

## 和 gpt-6 会诊（怎么发起）

```bash
ops/tools/consult.sh --list                  # 看历史 9 次会诊（主题 + 回复首行）
ops/tools/consult.sh <brief.md> 10           # 发起第 10 次（后台，15~25 分钟）
ops/tools/consult.sh --check 10              # 查是否写完
ops/tools/consult.sh <brief.md> 10 --wait    # 前台等（带进度）
```

- 走 `pi --print --provider lmstudio --model gpt-6-astra`（单向调用，回复落盘）；
  thinking 档位**只能是 low/medium/high/xhigh/max**（`minimal`/`off` 会被端点 400 拒绝）。
- 每次调用都强制附上"只读、不许重启/停/重配服务、不许跑 GPU 重负载基准、英文、量化"的 system prompt。
- 产物在 `ops/consultations/`（`brief-N.md` / `reply-N.md` / `reply-N.err` / `reply-N.rc`）。
- 完整方法论（何时该问、brief 骨架、结论如何回填 `OPS.md`）见仓库根 **`AGENTS.md` 第 2 节**。

## 子目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `bench/` | `warmup.py`（新鲜 token 预填计时）、`dec_bench.py`（解码+接受率+步时）、`prefill_ab.py`、`prefill_step.py`、`gridsweep.py`、`footprint.py` | **在用** |
| `tools/` | `forward8000.py`（:8000→:9393 转发，纯标准库）、`fdl.py`/`dl_blobs.py`/`dl_model.py`（断点续传下载）、`verify_ckpt.py`、`verify_sha256.py`、`resume.sh`、`vllm_logs.sh`、`analyze_trace.py`、`stage_parse.py`、**`consult.sh`（和 gpt-6-astra 会诊）** | 备用 |
| `patches/` | `qsa-alloc-heal.patch`（本机内存治愈补丁，反向应用已校验） | 重要 |
| `measurements/` | `perf-history.csv`、`bench-*.log`（新）与历史证据 json/log | 留档 |
| `prof/` | torch profiler 的 kineto trace | 留档 |
| `logs/` | Docker 时代的下载/安装日志（fdl.log、blobs*.log、pull*.log、install.log） | 留档 |
| `legacy-docker/` | `run_container.sh`（Docker 回退路径，**镜像已删、需先 `docker pull`**）、`IMAGE-PROVENANCE.md`（还原凭据）、`qsa_ops_heal.py`、`triton_cache/`（135 MB，Docker 路径用）、`junk/` | 回退用 |
| `consultations/` | 和 gpt-6-astra 的 9 次会诊原件：`brief-NN.md` / `reply-NN.md` / `.err` / `.rc`（从 `/tmp` 归档，2026-09-29） | 决策留档 |
| `diagnostics/` | 曾经的探针与最小复现（`minrepro*`、`*_instr.py`、`ablation.sh`、`allocprobe.sh`、`ladder.sh`、`vmm_probe.py`…） | ⚠️ **已退役，不要跑** |

## ⚠️ 已退役的探针（不要重跑）

`diagnostics/` 里跑 CUDA 内核的探针（`*_battery`、QXPROBE/QXGATE 路径）会让引擎崩一次：
引擎 CUDA abort → 主机 `nvlddmkm` 报 GPUID 错误 → dxgkrnl 生成 WATCHDOG 活转储（整机卡住数十秒）
+ WSL 写 1~5 GB 崩溃转储。因果链证据见 `ops/OPS.md` 9.20/9.21。**只读它们，不要执行。**

## 回退到 Docker

**镜像本体已于 2026-09-29 删除**（释放 25 GB；原生部署不依赖它，`vllm-native/` 零外部软链接）。
所以要回退时**先按 digest 重新拉取**（镜像仍在 Docker Hub，匿名可拉，22 层 / 8.61 GiB 压缩）：

```bash
docker pull 18gogogo/170hx1-qwen38nextf@sha256:9d8f3babf23360d46009f9c17a6747ce4c953e67169ffc4f9071a7ae5c338bd3
docker image inspect 18gogogo/170hx1-qwen38nextf:sm80 --format '{{.Id}}'   # 应等于上面的 digest
bin/stop.sh && ops/legacy-docker/run_container.sh                          # 约 6 分钟，端口/模型名/参数一致
```

完整的还原凭据（Image ID / RepoDigest / 22 层 digest 全列表）在
**`ops/legacy-docker/IMAGE-PROVENANCE.md`**：拉完后比对 Image ID 即可确认与当初完全同一份。
容器 `hong-pc` 也已删除（它只是那个镜像的实例，可写层里只有 `.humming` 编译缓存与 vllm 统计文件，无独有产物；
`run_container.sh` 本来就会 `docker stop -t 30` + `docker rm -f` 后重建同名容器）。

回退仍需要的宿主文件（都还在）：`run_container.sh`、`qsa_ops_heal.py`（内存治愈补丁）、
`triton_cache/`（Docker 路径用的热缓存，135 MB）；`*_PATCH` 变量无默认值，仅在诊断时手动传入。

## 测量与宿主内存的注意事项（会影响结论）

1. **解码对比必须"充分预热 + 宿主低压"**：预热 ≥2500 解码 token 且 `vmmemWSL < 32 GB`
   （先 `bin/drop_host_cache.sh` 并等 20~60 s 让 balloon 缩回）。同一二进制在高压下步时
   可虚高 25~40 %（17.4 ms → 22.2 ms），足以把 MTP=2 误判为"MTP=1 更慢"。
2. **`bench_prefill.py` 的 seed 固定**：换 seed 才有新鲜（未缓存）预填；重复同一 prompt 会命中
   前缀缓存（32K 预填 10.4 s → 0.68 s），那不是预填性能。
3. **每次加载模型，WSL 会攒 ~50 GB 干净页缓存**，`vmmemWSL` 涨到 55~62 GB、Windows 只剩 5~7 GB，
   整机发卡且解码变慢；`bin/start.sh` 会自动回收（PLE 走 `O_DIRECT`，不受影响）。
4. 解码吞吐与内容相关：`tok/s = (1 + 接受的 draft 数) / 步时`，接受率随内容在 50~80 % 波动。
