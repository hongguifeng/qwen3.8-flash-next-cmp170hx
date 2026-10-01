# AGENTS.md — 本仓库的 agent 工作手册

两件事最重要：**(A) 卡住时先和 gpt-6-astra 会诊（第 2 节，含可直接复制的命令）**；
**(B) 动服务/测量/长上下文之前必须遵守第 3 节的铁律**，否则会拿到假结论或把整机搞卡。

---

## 1. 项目速览

- **目标**：在一张 CMP 170HX（SM80，64 GiB）上跑 `Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`，
  95.4 GiB 的 BF16 PLE 表放在 SSD 上（`O_DIRECT` + 原生 AIO），上下文 262 144。
- **部署**：WSL2 内**原生 vLLM**（不再用 Docker 容器）。Docker 只作为回退路径，
  且**本机镜像已删除**（2026-09-29 释放 25 GB），回退前需先按
  `ops/legacy-docker/IMAGE-PROVENANCE.md` 里的 digest 重新 `docker pull`。
- **目录**：
  - `config/engine.env` = **端口与全部引擎参数的唯一默认值来源**（改端口只改这里，见第 4 节）
  - `vllm-native/` = 运行时（venv + 已打补丁的 vLLM 源码 + Triton 缓存 + 日志）
  - `bin/` = 日常入口：`start.sh` / `stop.sh` / `status.sh` / `logs.sh` / `bench.sh` / `drop_host_cache.sh`
  - `ops/` = 工具、证据、补丁、Docker 回退件、`OPS.md`（全部排查记录）
  - `docs/RESULTS-WSL2.md` = 本机性能数据；`docs/SCRIPTS.md` = **脚本总览/调用链/新增脚本约定**；
  `docs/DEPLOY-WSL2.md` = **部署与恢复手册**（前置条件、8 步部署、验收基线、故障排查表）；`results/pass3-wsl2-native/` = 原始 JSON
- **接口**：`http://127.0.0.1:8000/v1`（引擎直接监听 8000），模型名 `Qwen3.8-Flash-Next`。
  历史上曾用 9393 + 一个 `:8000→:9393` 转发器，2026-10-01 已把端口统一为 8000 并删除转发层（`ops/OPS.md §9.32`）。

---

## 2. 与 gpt-6 讨论（会诊）：使用场景与方法

### 2.1 什么时候该会诊（触发条件）

**该用**：
- 同一个卡点**连续失败 ≥2 次**，或意识到自己开始"猜着试"；
- 现象自相矛盾、或测量结果**不稳定/复现不了**；
- 要设计**决定性实验**，或要判断"某假设是否已被否证"；
- 将要执行**代价高或可能把机器搞坏/搞卡**的操作（重启、驱动级实验、长上下文压测）之前，需要外部评审；
- 需要一个**判读矩阵**（"若结果 A 则假设 1 成立、若 B 则假设 2"）来决定下一步。

**不该用**：读文档/读源码/跑一条命令就能确认的事；纯查询类问题。

历史情况：本项目一共会诊 **9 次**（`ops/consultations/`，原始 brief 与回复都在），
每一次都直接改变了排查方向（见 `ops/OPS.md` §8.1、§9.11–§9.18）。

### 2.2 怎么用（已验证的命令）

```bash
cd /home/hong/code/qwen3.8-flash-next-cmp170hx
nohup timeout 2400 pi --print \
  --provider lmstudio --model gpt-6-astra \
  --thinking high --approve \
  --append-system-prompt "You are a read-only reviewer. Do NOT restart, stop, or reconfigure the running service or the host; no GPU-heavy benchmarks. Answer in English, concretely and quantitatively, ranked by expected value per unit cost." \
  "$(cat /tmp/astra_briefN.md)" > /tmp/astra_replyN.md 2>/tmp/astra_errN.log &
```

**一键脚本（推荐）**：

```bash
ops/tools/consult.sh --list                  # 看历史会诊（本项目已做过 9 次，原件在 ops/consultations/）
ops/tools/consult.sh <brief.md> 10           # 发起第 10 次：自动加约束、后台跑、立即返回
ops/tools/consult.sh --check 10              # 是否写完（读 reply-10.rc / 字节数 / 首行）
ops/tools/consult.sh <brief.md> 10 --wait    # 前台等（带进度）+ 结果汇报
```

脚本自动加约束、后台调用、归档 brief，产物在 `ops/consultations/`：`brief-N.md` / `reply-N.md` /
`reply-N.err` / `reply-N.rc`（`.rc` 内容 = 真实退出码，作为完成标记）。
可调：`CONSULT_PROVIDER` / `CONSULT_MODEL` / `CONSULT_THINKING` / `CONSULT_TIMEOUT`。

实测确认的要点（2026-09-29，`pi 0.85.1`）：

| 事项 | 事实 |
|---|---|
| 可用 provider/model | `lmstudio gpt-6-astra`（250K 上下文）← 一直用这个；备选 `openrouter openai/gpt-6-astra`（1.1M） |
| thinking 档位 | 只有 `low / medium / high / xhigh / max`。**`minimal`/`off` 会被端点拒绝**（HTTP 400） |
| 调用形式 | `--print` 是**单向**的：发出去、回完、退出。回复只在 stdout ⇒ **必须重定向到文件再读** |
| 耗时 | 单次 **15~25 分钟**（#9 用 17 分钟）⇒ 一律 `nohup … &` 后台跑，`timeout 2400` 起 |
| 读回复 | `read /tmp/astra_replyN.md`（别 `cat` 刷屏）；判断结束看文件是否还在增长（`wc -c`）或进程是否退出 |
| 会话 | 加 `--no-session` 不留会话记录；要接着上次聊用 `--continue`/`--session <id>` |
| 查模型 | `pi --list-models <关键词>`；`pi auth` 查 provider 凭据状态 |
| 注意 | 本文件（`AGENTS.md`）会被 pi 自动加载进会话；给顾问的 brief 若不需要仓库上下文，可加 `--no-context-files` 省 token |

### 2.3 brief（提问文件）怎么写 —— 实测最有效的结构

按 `ops/consultations/brief-09.md` 的骨架，逐节填：

1. **标题**：`# Consultation #N — <本轮要解决的一句话>`
2. **项目背景一段话**：模型/引擎/平台/部署方式（够用即可，不要长篇）
3. `## Established facts (unchanged)`：**已经用数据确立**的事实，每条带数字与测量方法，
   明确"这些不用再讨论"，避免重复历史。
4. `## NEW result #k — …`：本轮新证据。每条都要有：怎么测的、数字、以及**它否证了谁**。
5. `## What I want from you`：编号提问 (a)(b)(c)…，并明确要求：
   - 具体、可量化、**按性价比排序**；
   - 给出**最便宜的决定性实验**（含能在两种环境下同样运行的版本）；
   - 给出**判读矩阵**：哪种结果支持哪个假设，哪种结果是否证。
6. `## Constraints`：不能做什么（见 2.4）。
7. **原始材料**：关键源码片段、日志、命令输出——**贴原文**，不要转述。

### 2.4 必须写进 system prompt 的约束（当年每次都写）

> read-only inspection only；**不要**重启/停止/重配正在跑的服务或主机；
> **不要**跑 GPU 重负载基准；用英文回答，具体、可量化、按期望收益排序。

理由：顾问的建议常常是"重启一下试试/跑个压测"，而这个服务的重启代价 4~5 分钟、
内核级实验会让整机冻结数十秒。把约束写死在 system prompt 里最省事。

### 2.5 拿到回复后怎么用

- 回复与 brief 一起归档到 `ops/consultations/`（**不要留在 `/tmp`**，会被清掉）。
- 在 `ops/OPS.md` 里写清三件事：**它判定什么 → 我实测什么 → 因此哪个假设成立/被否证**。
  只写"问了 gpt-6"没有价值；`OPS.md` §9.11–§9.18 的写法就是范例（含"被否证的假设"清单）。
- 它给的实验做完后**回填判读矩阵**：命中哪一行、否证了哪个候选、下一条最便宜的路是什么。
- 若它的建议与实测冲突，以**实测**为准，并把冲突记进 OPS.md（历史上发生过，见 §9.12/§9.14）。

---

## 3. 铁律（不看会出事）

1. **每次改动都要以"可用且已验证"结束**：`bin/status.sh` 显示 `health=200`，并跑一次真实预填测量。
2. **别用 `localhost`**：Windows 侧必须 `127.0.0.1` —— mirrored 网络不转发 IPv6 回环。
   （容器时代能用 `localhost` 是因为端口由 Windows 侧发布。）
3. **测解码必须"预热 ≥2500 token + `vmmemWSL < 32 GB`"**，否则步时虚高 25~40%
   （同一二进制：17.4 ms ↔ 22.2 ms），足以把 `MTP=2` 误判成"更慢"。先 `bin/drop_host_cache.sh`。
4. **`benchmarks/bench_prefill.py` 每次换 `--seed`**：seed 固定 ⇒ prompt 与上轮相同 ⇒
   命中前缀缓存（32K 预填 10.4 s → 0.68 s），那不是预填性能。
5. **跑过 ≥96K 预填后检查是否"中毒"**：后置全新 2048 预填应 ≈0.68 s。治愈线程
   `QSA_ALLOC_HEAL=1` 不要关（关掉后长上下文会让 prefill 慢 4.5×、decode 慢 1.7×，且必须重启才恢复）。
6. **引擎内 CUDA 探针永久退役**（`*_battery` / QXPROBE / QXGATE）：会让引擎崩溃并触发主机
   `dxgkrnl` WATCHDOG 活转储（整机卡死数十秒）+ 写入 1~5 GB 崩溃转储。历史证据见 `ops/OPS.md` §9.20/§9.21。
7. **每次加载模型，WSL 会攒 ~50 GB 干净页缓存**（`vmmemWSL` 涨到 55~62 GB，Windows 只剩 5~7 GB），
   会让整机发卡且解码变慢 ⇒ 跑 `bin/drop_host_cache.sh` 还内存（不影响引擎，PLE 走 `O_DIRECT`）。
8. **停服务前先 `bin/stop.sh --check`**（会看 `num_requests_running`）；不要 `kill -9`，用 `bin/stop.sh` 优雅停。
9. **重启 = 4~5 分钟不可用**，先征得用户同意；`.wslconfig` 改动要 `wsl --shutdown` 才生效（也要先问）。
10. **`~/vllm` 已废弃**：所有工具在 `ops/`，不要再往 `~/vllm` 写东西。
11. **Docker 镜像已删（2026-09-29）**：`18gogogo/170hx1-qwen38nextf:sm80` 本机已不存在，
    原生路径（`vllm-native/`，零外部软链接）不依赖它。要回退 Docker 必须先
    `docker pull 18gogogo/170hx1-qwen38nextf@sha256:9d8f3bab…`（凭据/校验见
    `ops/legacy-docker/IMAGE-PROVENANCE.md`），再 `bin/stop.sh && ops/legacy-docker/run_container.sh`。
12. **主机 NVMe 会掉盘（2026-09-29 已发生一次）**：D: 是 KIOXIA EXCERIA G2 NVMe（项目全部运行时
    `D:\wsl\ext4.vhdx` 841 GiB + 143 GB 模型都在它上面）；掉盘会连 WSL/引擎一起带崩（蓝屏 `0x7A`）。
    ⇒ **引擎在跑时不要做 GB 级删除（docker rmi/大清理）或全盘 `du`/`find` 扫描**，housekeeping 一律先 `bin/stop.sh`。
    事件与证据见 `ops/OPS.md` §9.30；盘已是最新固件 ECFA17.3，优先考虑关 ASPM/PCIe 链路电源管理。

---

## 4. 常见任务速查

```bash
./start.sh                   # 启动（根目录入口，薄封装 bin/start.sh；等就绪 250~320 s）
bin/start.sh                 # 同上（等价）；--wait/--keep-cache/--foreground 都支持
./deploy.sh check            # 部署/环境自检（只读）：环境/GPU/内存/磁盘/模型/运行时/补丁/服务
./deploy.sh verify           # 验收：health=200 + 新鲜 2048 预填对比基线（判读见 docs/DEPLOY-WSL2.md §5）
bin/status.sh                # 健康 / 请求 / 显存 / 宿主内存 / MTP 接受率（--watch 5 刷新、--short 单行）
bin/logs.sh -f               # 跟踪日志（-e 错误 / --startup 启动行 / --heal 内存治愈 / --list 归档）
bin/bench.sh                 # 体检 + 自动存档到 ops/measurements/perf-history.csv
bin/stop.sh --check          # 只报告"现在停会不会打断用户请求"
bin/drop_host_cache.sh       # 把 WSL 页缓存还给 Windows
./start.sh --params          # 打印此刻生效的参数与来源（不启动任何东西）
vllm-native/bin/run_native.sh print-cmd   # 干跑：打印将执行的 vllm 命令行（验证用）
ops/tools/consult.sh <brief> # 和 gpt-6-astra 会诊（见第 2 节）
```

**改端口/改参数只在 `config/engine.env` 一处改**（不要再去改脚本里的默认值；那里是唯一来源，
`bin/*` 与 `vllm-native/bin/run_native.sh` 都 source 它，`ops/bench/*.py` 通过 `ops/bench/_cfg.py` 读它）：

```bash
$EDITOR config/engine.env    # 永久改默认值
QWEN_MTP=1 ./start.sh        # 临时覆盖一次（环境变量优先）
```

可调项：`QWEN_PORT`（8000）、`QWEN_MTP`（默认 2 = 实测最快）、`QWEN_GPU_MEMORY`（0.94）、
`QWEN_CONTEXT`（262144）、`QWEN_SEQS`（4）、`QWEN_BATCH_TOKENS`（**2048，改成 8192 会慢 2 倍**）。
脚本改动/新增脚本前先看 `docs/SCRIPTS.md`（含 4 条硬约定）。

**基线数字**（判异常用，详见 `docs/RESULTS-WSL2.md`）：
预填 2048 ≈ 0.64~0.73 s、8192 ≈ 2.0~2.4 s、131072 ≈ **48.7~49.0 s**
（2026-10-01 在功耗上限 220 W 下复测；9 月旧记录是 54.6 s / 54~56 s）；
解码步时 ≈ 15.2~15.7 ms（MTP=1）/ 17.2~17.6 ms（MTP=2）；稳态解码 ≈ 111 / ≈126 tok/s。

**当前 GPU 设置**（2026-10-01 起，`nvidia-smi` 实测 + `ops/OPS.md §9.33`）：
功耗上限 **220 W**（default 250 / max 300）、显存时钟 **1728 MHz 恒定**（= 硬件上限）、
SM 负载峰值 1485 MHz（上限 1695）、Windows 驱动 KMD **610.88**。
解码忙时实测只用 177 W（均值）/ 204 W（峰值）⇒ 功耗上限只影响长预填的瞬时峰值。
