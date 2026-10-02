# SCRIPTS.md —— 脚本总览与调用链（本项目的"哪个脚本管什么"）

> 背景：2026-10-01 之前，端口默认值散落在 5 个地方，`bin/_common.sh` 里写的是 **8001**、
> 引擎启动器写的是 **9393**，导致 `bin/status.sh` 报 `health=000`、
> `bin/start.sh` 的"已在运行"守卫失效。本次整理后**默认值只剩一处**。
>
> 2026-10-01 后续调整：**默认端口 = 8000，引擎直接监听 8000**；
> 原来的 `:8000 → :9393` 转发层（`ops/tools/forward8000.py`）已删除，不再需要。

---

## 1. 唯一默认值来源：`config/engine.env`

```bash
config/engine.env        # 端口、模型、上下文、MTP、PLE offload、治愈线程……全在这里
```

* 被 `bin/_common.sh`、`vllm-native/bin/run_native.sh`、`ops/tools/metrics_web.sh` source；
  `ops/bench/*.py` 通过 `ops/bench/_cfg.py` 读它。
* 写法统一为 `: "${QWEN_PORT:=8000}"` ⇒ **环境变量永远优先**，文件只是默认值。
* 里面不放逻辑、不执行命令。

**改端口 / 改参数只有两种做法，都不需要动别的文件：**

```bash
$EDITOR config/engine.env        # ① 永久改默认值（含加模型档位）
QWEN_MTP=1 ./start.sh            # ② 临时覆盖一次（不改文件）

./start.sh --params              # 看此刻实际生效的值与来源
./start.sh --model list          # 看有哪些模型档位（模型目录 / 端口 / 显卡）
```

### 1.1 模型档位（variant）：选模型 + 选卡

**一个档位 = 一套「模型目录 + 对外模型名 + 端口 + 显卡」**。默认两个：

| 档位 | 模型 | 端口 | 显卡 | pid / 日志 |
|---|---|---|---|---|
| `main` | `Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` | 8000 | GPU0 | `server.pid` / `server.log` |
| `unc` | `Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP`（别名 `uncensored`/`b`） | 8001 | GPU1 | `server-unc.pid` / `server-unc.log` |

```bash
# 最常用：位置参数就够了（第一个 = 档位，第二个 = 显卡）
./start.sh unc                     # 起档位 unc（默认 GPU1 / :8001）
./start.sh unc 0                   # 档位 unc + GPU0（端口仍是该档位默认的 8001）
./start.sh 1                       # 只换卡，模型仍是 main
./start.sh list                    # 列出所有档位（不启动）
# 长写法 / 环境变量（完全等价）
./start.sh --model unc --gpu 0
QWEN_MODEL=unc ./start.sh
# 管哪个档位就带同一个参数（stop/status/logs/bench 都支持）
bin/status.sh --model unc          bin/stop.sh --model unc          bin/bench.sh --model unc --full
```

两个实例可以**同时跑**（不同卡/端口/pid/日志，互不覆盖）；`start.sh` 启动前会检查目标端口与目标显卡的占用。

**新增一个模型档位（只改 `config/engine.env`，脚本不用动）**：

```bash
: "${QWEN_MODELS:=main unc trl}"                    # ① 档位清单（第一个是默认档位）
: "${QWEN_TRL_MODEL_DIR:=/home/hong/models/…}"      # ② 该档位的四个变量
: "${QWEN_TRL_SERVED_NAME:=Qwen3.8-Flash-Next-TRL}"
: "${QWEN_TRL_PORT:=8002}"
: "${QWEN_TRL_GPU:=0}"
```

优先级：`--model`/`QWEN_MODEL` > 显式环境变量（`QWEN_PORT=` / `CUDA_VISIBLE_DEVICES=`）> 档位默认值。
历史名字（2026-10-01 §9.37 用的那套）继续有效：`QWEN_INSTANCE=b` ≡ `QWEN_MODEL=unc`，`QWEN_B_*` 仍然能覆盖档位 unc。

---

## 2. 调用链（`./start.sh` 到底走了谁）

```
./start.sh                              根入口，薄封装（24 行，只有 exec 转发）
  └─ exec bin/start.sh                  真正的启动逻辑（等就绪 / 回收内存 / 拉转发器）
       ├─ source bin/_common.sh         ← 加载 config/engine.env + 公共函数
       │     └─ LAUNCHER=vllm-native/bin/run_native.sh
       └─ "$LAUNCHER" start             ← 唯一给 `vllm serve` 拼命令行的地方
            └─ setsid nohup vllm serve …  引擎脱离会话，脚本退出后继续跑
```

其它命令同构：`bin/stop.sh` / `bin/status.sh` / `bin/logs.sh` / `bin/bench.sh`
都 `source bin/_common.sh` 取得同一套端口与路径，因此**不可能再出现端口不一致**。

---

## 3. 脚本清单

### 3.1 日常入口（活跃，认准这一组）

| 脚本 | 职责 | 常用 |
|---|---|---|
| `./start.sh` | 根入口，等价于 `bin/start.sh` | `./start.sh`、`--model unc`、`--gpu 1`、`--params` |
| `./deploy.sh` | 根入口 → `ops/deploy/deploy_wsl2.sh`：**部署/自检**（默认只读） | `check`、`check --json`、`verify`、`plan`、`install --yes` |
| `bin/start.sh` | 启动 + 等就绪 + 回收宿主内存；支持**档位/显卡**选择 | `--model unc`、`--gpu 1`、`--model list`、`--wait 900`、`--foreground` |
| `bin/stop.sh` | 优雅停止（SIGTERM，60 s；**默认绝不 SIGKILL**） | `--check`（先跑这个！）、`--model unc` |
| `bin/status.sh` | 健康/进程/请求/KV/MTP 接受率/宿主内存 | `--short`（脚本友好）、`--watch 5`、`--model unc` |
| `bin/logs.sh` | 看日志（错误/启动/治愈/归档） | `-f`、`-e`、`--startup`、`--heal`、`--clean`、`--model unc` |
| `bin/bench.sh` | 体检 + 永久存档到 `ops/measurements/perf-history.csv`（非 main 档位的 tag 自动带 `@档位`，日志名带档位） | `--full`、`--reps 3`、`--tag x`、`--model unc` |
| `bin/drop_host_cache.sh` | 把 WSL 页缓存还给 Windows（不碰引擎） | 加载模型后必跑 |
| `vllm-native/bin/run_native.sh` **选档位** | 直接用环境变量选：`QWEN_MODEL=unc` ⇒ 模型目录/端口/卡/pid/日志全部跟着变（`server-unc.*`），**不影响主实例** | `QWEN_MODEL=unc vllm-native/bin/run_native.sh start\|status\|stop\|print-cmd`（见 `ops/OPS.md §9.37`、§9.40） |

### 3.2 引擎层（只有引擎自己用）

| 脚本 | 职责 |
|---|---|
| `vllm-native/bin/run_native.sh` | **唯一**直接调用 `vllm serve` 的脚本：`start/stop/restart/status/print-cmd/params/foreground` |
| `config/engine.env` | 参数默认值（唯一来源） |
| `ops/deploy/deploy_wsl2.sh` | 部署/自检：`check`（只读体检，含端口一致性比对）/`plan`/`install`/`start`(→`bin/start.sh`)/`verify`/`all`；文档见 `docs/DEPLOY-WSL2.md` |

`deploy.sh check` 的检查项名字就是文档里的锚点（`host.*`、`disk.space`、`model.*`、`runtime.*`、`patch.*`、`ple.lib`、`cache.triton`、`config.*`、`service.health`），
FAIL 时每条都带「→ 修复」；`deploy.sh verify` 用 `ops/bench/warmup.py`（新鲜 token id）取 2048 稳态预填并对比基线。

新增的 `print-cmd` 是**干跑**：只打印将执行的命令行，不启动任何东西。
改完脚本拿它和线上进程逐项比对，是验证"重构没改语义"的最便宜办法：

```bash
diff <(tr '\0' '\n' < /proc/$(cat vllm-native/logs/server.pid)/cmdline | tail -n +2) \
     <(./vllm-native/bin/run_native.sh print-cmd)
```

### 3.3 测量 / 工具（活跃）

| 脚本 | 职责 |
|---|---|
| `ops/bench/warmup.py` | 冷预填测量（全新 token id，永远不吃前缀缓存） |
| `ops/bench/dec_bench.py` | 解码步时 / 接受率（流式，剔除 TTFT） |
| `ops/bench/prefill_ab.py` / `prefill_step.py` | 预填 A/B、并发预填吞吐 |
| `ops/bench/_cfg.py` | 给上面这些探针提供"引擎地址"（读 config/engine.env） |
| `ops/tools/metrics_web.sh` / `.py` | `/metrics` 只读看板（默认 `127.0.0.1:9494`，上游端口随 `QWEN_PORT`） |
| `ops/tools/metrics_web.html` / `dashboard.js` | 看板的**页面与渲染**（零依赖手写：KPI 条 + 差分表 + 曲线/进度条 + 深浅主题） |
| `ops/tools/consult.sh` | 与 gpt-6-astra 会诊（见 `AGENTS.md` 第 2 节） |
| `ops/tools/fdl.py` | 可续传并行下载器（分块 + Range + fsync + **稀疏洞检测**）；仓库/版本/目标目录可用 `REPO`/`REVISION`/`DEST`/`CONCURRENCY` 覆盖 |
| `ops/tools/inspect_model_dir.py` | 模型目录**离线体检**：文件齐全/尺寸、每个 shard 的 header+尾部自洽、**洞（st_blocks）检测**、PLE 布局，`--ref` 可与现役模型比对 |
| `ops/tools/cmp_model_dirs.py` | 两个模型目录逐张量**采样字节比对**（上游不给 SHA-256 时唯一的内容级证据），按类别给出"改了哪些部分" |
| `benchmarks/bench_prefill.py` | 手动预填压测；**每次必须换 `--seed`**（否则命中前缀缓存） |
| `ops/bench/codebench.py` | **代码能力 A/B 基准**（HumanEval / HumanEval+ / MBPP-san，执行式判定）：`run --dataset … --base-url … --out x.json` 跑一个引擎，`compare a.json b.json` 出 pass@1 + 四格表 + McNemar。协议两边必须一致，详见 `ops/OPS.md §9.37` |

**看板页面怎么改**：`metrics_web.py` 每次请求都从磁盘重读 `metrics_web.html` / `dashboard.js`
（响应头 `Cache-Control: no-store`）⇒ **改完页面不用重启看板进程，浏览器刷一下就行**；
只有改端口/上游地址才需要 `ops/tools/metrics_web.sh stop && start`。
页面约定：数字包在 `<span class="n">`（nowrap+等宽），长注释单独一行 `.note`，
两者混排会把数字断成两行（2026-10-01 重做显示就是因为这个）。

### 3.4 遗留 / 请勿混用（Docker 时代）

这些属于**回退路径或历史诊断**，不与原生路径共享配置，**不要在原生流程里用**：

| 位置 | 说明 |
|---|---|
| `ops/legacy-docker/run_container.sh` | Docker 回退（镜像已删，需先按 `IMAGE-PROVENANCE.md` 重新 pull） |
| `ops/diagnostics/*` | 历史诊断探针：硬编码 `localhost:9393` / `PORT=9393`，且多数直接调 `run_container.sh`（Docker 路径）；`localhost` 还违反本仓铁律 2 |
| ~~`ops/tools/forward8000.py`~~ | **已删除**（2026-10-01）：引擎直接监听 8000 就不需要转发层了。删因与源码见 `ops/OPS.md §9.32` |
| `ops/tools/vllm_logs.sh` | 看 **Docker 容器**日志的旧工具（原生路径请用 `bin/logs.sh`） |
| `ops/bench/deepswe/*` | DeepSWE v1.1 试点（`run_pilot.sh` + `patch_pier_egress.py`）：**2026-10-02 用户决定放弃，已连同产物全部删除**（Pier 已卸载、squid 补丁已还原）。要重做请看 `ops/OPS.md §9.38`（含可行性实测、思考强度矩阵、两次蓝屏取证） |
| `scripts/`、`vllm-native/opt/` | 镜像内 payload / Docker 构建脚本（`serve.sh` 默认端口 8000），保持原样以便回退 |

> 保留它们的硬编码是**有意的**：那是另一个（容器）环境的配置，改了反而会让回退路径失效。

---

## 4. 新增脚本必须遵守的约定

1. **不要写默认值**：端口/模型/参数一律从 `config/engine.env` 取（bash 就 `source bin/_common.sh`，
   Python 就 `import _cfg`）。确实需要兜底时，兜底值也要指向同一个来源。
2. **help 用 `usage`/`help_exit`**：`-h|--help) help_exit "$0" ;;`
   —— 它把文件头部注释块原样打印，**别再用 `sed -n '2,17p'` 数行号**（加一行注释就错位）。
3. **别自己另起一套 ROOT/PORT 解析**：`source bin/_common.sh` 后 `$ROOT`/`$BASE_URL`/`$LOG_FILE` 都在。
4. **不重启、不强杀**：停服务一律走 `bin/stop.sh`（它委托 `run_native.sh stop`）；需要 SIGKILL
   才显式 `QWEN_FORCE=1`。任何改动都要以 `bin/status.sh` 显示 `health=200` 收尾。

---

## 5. 本次整理修复的历史 bug（不是为了好看，都是真会咬人的）

| 位置 | 原状 | 后果 |
|---|---|---|
| `bin/_common.sh:15` | `PORT` 默认 **8001**（引擎是 9393） | `status.sh` 永远 `health=000`；`start.sh` 的"已在运行"守卫失效 ⇒ 会起**第二个引擎**抢显存 |
| `bin/stop.sh` | 收尾查的是 8001 | 打印"端口已释放 ✔"是假的 |
| `bin/logs.sh --clean` | 注释说"归档并清空"，实现是 `: > log` | **静默丢掉全部日志**；现在先 `cp` 归档再截断（引擎 fd 不受影响） |
| `run_native.sh stop` | 60 s 后自动 SIGKILL | 与铁律 8 冲突；现在默认报错退出，除非 `QWEN_FORCE=1` |
| `run_native.sh` 修缓存提示 | 路径算错（`$ROOT/..`） | 提示里给出的是不存在的目录 |
| `bin/bench.sh` | 硬编码 `127.0.0.1:9393` | 改端口后体检会打空 |
| 所有脚本 `--help` | `sed -n '2,Np'` 数行号 | 加/删注释后 help 错位 |
| `ops/tools/forward8000.py` | 静默（`forward.log` 一直 0 字节）且端口写死 | 转发器挂了没有任何线索 |
