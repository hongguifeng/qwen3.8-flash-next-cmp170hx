# DEPLOY-WSL2.md —— 在 WSL2 上部署 / 恢复本服务（本机实测版）

> **这份文档只讲"怎么把服务装起来、验起来、恢复起来"**，而且描述的是**当前这台机器真实在跑的那套环境**，
> 每个数字都是实测（来源标在表里），不是估算。
>
> 配套脚本：`./deploy.sh`（根入口）→ `ops/deploy/deploy_wsl2.sh`
>
> | 想查什么 | 去哪 |
> |---|---|
> | 工作手册 + 12 条铁律 + 会诊流程 | `AGENTS.md` |
> | 脚本调用链 / 新增脚本约定 | `docs/SCRIPTS.md` |
> | 性能数据与复现命令 | `docs/RESULTS-WSL2.md` |
> | 全部排查记录（踩过的坑、证据链） | `ops/OPS.md` |
> | 上游原始安装说明（英文，Docker 时代） | `docs/GUIDE.md` |
>
> **一句话**：机器已部署好 ⇒ 平时只要 `./start.sh`；开机/`wsl --shutdown` 后 ⇒ `./deploy.sh check && ./start.sh`；
> 换机器或重装 ⇒ 按 §4 的 8 步走，每步都有验证命令。

---

## 0. TL;DR —— 三种场景

| 场景 | 命令 | 说明 |
|---|---|---|
| 服务崩了/想让服务起来（已部署） | `./start.sh` | 等 `health=200`（首次加载权重约 250~320 s）后自动回收 Windows 页缓存 |
| 重启电脑 / `wsl --shutdown` 之后 | `./deploy.sh check && ./start.sh` | 引擎**没有**开机自启（用户明确不要）；check 会告诉你缺什么 |
| 换一台新机器 / 完全重装 | `./deploy.sh plan` 然后按 §4 走 | `./deploy.sh install --yes` 能幂等补齐目录/补丁/`.so`；权重需 `--download-model` |
| 怀疑状态不对 | `./deploy.sh verify`（加 `--full` 跑到 131072） | health + 新鲜 2048 预填对比基线，给出 PASS/WARN/FAIL 与修复建议 |
| 服务在跑，想知道能不能动它 | `bin/stop.sh --check` | 只看 `num_requests_running`，不会动手（铁律 8） |

---

## 1. 本机环境快照（2026-10-01 实测）

| 项 | 实测值 | 来源 |
|---|---|---|
| Windows | 11 24H2 (26100.4061)，物理内存 88 GB，16 物理核 | `docs/RESULTS-WSL2.md`、Windows 侧 |
| WSL | **2.7.11.0**，内核 **6.18.33.2-microsoft-standard-WSL2**，发行版 Ubuntu 22.04.5 LTS | `wsl.exe --version`、`/proc/version` |
| 容器资源 | `.wslconfig`：`processors=16`、`memory=48GB`（`free -g` 显示 47 GiB 可用） | `/mnt/c/Users/<你>/.wslconfig` |
| 网络模式 | `networkingMode=mirrored`（⇒ Windows 侧可用 `127.0.0.1`） | 同上 |
| GPU | NVIDIA CMP 170HX，**65 536 MiB**，SM80，枚举到两张（`0B:00.0` 在用 / `0C:00.0` 空闲），引擎只用 `CUDA_VISIBLE_DEVICES=0` 那张 | `nvidia-smi.exe --query-gpu=…` |
| 驱动 | KMD **610.88** / NVIDIA-SMI 610.57.01 / CUDA UMD 13.3（2026-10-01 实测；9 月记录是 KMD 616.92 / SMI 615.71.08 / UMD 13.4，已在 §2.1 说明现在的判据） | `nvidia-smi` |
| GPU 时钟/功耗 | 空闲 SM 210 MHz，负载峰值 **1485 MHz**（上限 1695）；**功耗上限 220 W**（requested 220 / default 250 / max 300）；显存 **1728 MHz 恒定**（= `Max Clocks: Memory`，硬件上限）；空闲 33 °C | `nvidia-smi`，详见 `ops/OPS.md §9.33` |
| 磁盘 | 项目所在 ext4（`/dev/sdd`，即 `D:\wsl\ext4.vhdx`）1006 GiB，当前可用 216 GiB | `df -h` |
| 运行时 | 原生 vLLM **0.29.1rc1.dev402+ga5a30471f.ple1**（editable）、torch **2.13.0+cu130**、triton **3.7.1**、transformers 5.17.0、Python **3.12.14** | `./deploy.sh check` |
| 模型 | `Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`，**13 个 safetensors / 142.5 GiB** | `deploy.sh check → model.*` |
| 服务 | `http://127.0.0.1:8000/v1`，模型名 `Qwen3.8-Flash-Next`，上下文 262 144，MTP=2 | `config/engine.env` |

引擎加载后的实测形状（`bin/logs.sh --startup` 可见）：

```
Available KV cache memory: 8.69 GiB
GPU KV cache size: 307,041 tokens, Maximum concurrency for 262,144 tokens per request: 1.17x
init engine (profile, create kv cache, warmup model) took 44.50 s
```

---

## 2. 前置条件

### 2.1 Windows 侧（这些不满足，WSL 里怎么折腾都没用）

1. **Windows NVIDIA 驱动：能用即可，本机现为 KMD 610.88**：WSL 的 CUDA 走 `/dev/dxg`，由 **Windows 驱动**提供。
   **绝对不要在 WSL 里装 Linux 版 NVIDIA 驱动**（会把 `/usr/lib/wsl/lib` 搞坏）。
   ```bash
   /mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=name,memory.total,driver_version --format=csv
   # NVIDIA Graphics Device, 65536 MiB, 610.88   ← 2026-10-01 实测
   ```
   > 历史措辞曾是"≥ 616.92"（9 月那份环境的实录）。**2026-10-01 在 KMD 610.88 上完整复测过**：
   > 启动 294 s、健康检查 200、预填/解码全部达到或优于基线（`ops/OPS.md §9.33`）⇒ 610.88 已验证可用。
   > 换驱动只需确认 `/usr/lib/wsl/lib/libcuda.so*` 存在且 `nvidia-smi` 能枚举到 GPU，不必强求某个具体版本号。
2. **WSL ≥ 2.7.11**：`wsl.exe --version`；升级用 `wsl --update`（Windows 侧执行）。
3. **`.wslconfig`**（`%USERPROFILE%\.wslconfig`，本机实测内容）：
   ```ini
   [wsl2]
   processors=16
   memory=48GB              # guest 实际可用 47 GiB；引擎加载 143 GB 权重需要宿主留余量
   nestedVirtualization=true
   networkingMode=mirrored  # ⇒ Windows 侧必须用 127.0.0.1（铁律 2）
   autoProxy=true
   dnsTunneling=true
   firewall=true
   guiApplications=false    # 不需要 GUI
   ```
   注意两点：① 改完必须 `wsl --shutdown` 才生效（**先征得用户同意**，铁律 9）；
   ② 本机**刻意没有** `autoMemoryReclaim` 那一行（它曾导致宿主内存被回收后解码变慢，见 `ops/OPS.md §9.22.1`）。
4. **NVMe 稳定性**：`D:`（KIOXIA EXCERIA G2）承载 `D:\wsl\ext4.vhdx` 与全部运行时，2026-09-29 **掉过一次盘并蓝屏 `0x7A`**。
   ⇒ 建议在 Windows 侧关闭该盘的 ASPM / PCIe 链路电源管理；**引擎在跑时不要做 GB 级删除或全盘扫描**（铁律 12）。

### 2.2 硬件

* **GPU**：SM80（Ampere）且 **≥ 64 GiB 显存**。权重在显存里约 47 GiB（PLE 表 95.4 GiB 不在显存里，走 SSD），
  剩余给 KV cache 与 CUDA 图；本机 `gpu-memory-utilization=0.94` 时 KV cache 只有 8.69 GiB。
* **磁盘预算**：

  | 内容 | 大小 |
  |---|---|
  | 模型权重（含 95.4 GiB PLE 表分片） | 142.5 GiB |
  | 运行时 `vllm-native/`（venv 7 GB + 源码 958 MB + triton 缓存 158 MB + uv-python 110 MB） | ≈8.3 GiB |
  | 仓库本体（含 ops/ 证据与日志） | ≈150 MB |
  | **合计 + 余量** | **≥ 250 GiB 空闲** |
* **内存**：物理 88 GB / guest 48 GB。PLE 表走 `O_DIRECT` **不占页缓存**，但加载权重期间 WSL 会攒下 ~50 GB 干净页缓存，
  加载完必须 `bin/drop_host_cache.sh` 还内存（铁律 7；`./start.sh` 默认会做）。

### 2.3 WSL 内的软件

| 需要 | 用途 | 本机 |
|---|---|---|
| `git`、`curl`、`ss`、`stat` | 全部脚本的前置 | ✅ |
| `cc` / `gcc`（build-essential） | 编 `ple_ssd_io.so` | ✅ |
| `python3` | 只给 `deploy.sh` / 体检用 | ✅ |
| `uv` | **仅路线 B**（自己构建运行时）需要 | 本机走路线 A，未依赖 |
| Windows 侧 `powershell.exe`、`nvidia-smi.exe`、`wsl.exe` | `bin/status.sh` 读宿主内存/显存 | ✅（路径写死在 `bin/_common.sh`） |

---

## 3. 目录布局：什么放在哪

```
/home/hong/code/qwen3.8-flash-next-cmp170hx/
├── start.sh                    ← 启动入口（薄封装 → bin/start.sh）
├── deploy.sh                   ← 部署/自检入口（薄封装 → ops/deploy/deploy_wsl2.sh）
├── config/engine.env           ← 【唯一默认值来源】端口/模型/上下文/MTP/PLE offload/治愈线程
├── bin/                        ← 日常入口：start/stop/status/logs/bench/drop_host_cache/_common
├── vllm-native/                ← 原生引擎本体（大部分被 .gitignore 排除，见下表）
│   ├── bin/run_native.sh       ← 【唯一】拼 `vllm serve` 命令行的地方 ✅ 入库
│   ├── opt/entrypoint.sh       ← 镜像时代的等价入口（回退用）✅ 入库
│   ├── opt/download-model.sh   ← 钉住 revision 的官方下载脚本 ✅ 入库
│   ├── opt/vllm/.venv/         ← venv（Python 3.12.14，7 GB）❌ 可重建
│   ├── opt/vllm/src/           ← vLLM 源码树（958 MB，已打两个补丁）❌ 见 §4.4
│   ├── opt/vllm/optimization/ple_ssd_io.so   ← 16 KB，编出来 ❌ 见 §4.6
│   ├── uv-python/              ← venv 的基解释器（110 MB）❌ 可重建
│   ├── triton_cache/           ← 410 个已编译 kernel 目录（158 MB）❌ 可重建
│   └── logs/                   ← server.log / server.pid ❌ 运行数据
├── ops/                        ← 工具、证据、补丁、Docker 回退（ops/OPS.md 是排查总账）
│   ├── deploy/deploy_wsl2.sh   ← 本文件的脚本化（check/plan/install/start/verify）
│   ├── patches/qsa-alloc-heal.patch   ← 第二个补丁（分配器治愈线程）
│   └── tools/                  ← fdl.py（续传下载）、verify_ckpt.py、metrics_web.*、consult.sh
└── docs/                       ← 本文件、SCRIPTS.md、RESULTS-WSL2.md、GUIDE.md

/home/hong/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP/     ← 模型权重（不在仓库里）
```

**模型目录里那个 102.4 GB 的分片就是要紧的东西**：

| 文件 | 大小 | 说明 |
|---|---|---|
| `model-00001-of-00011.safetensors` | **102.4 GB** | **BF16 PLE 表**（95.4 GiB）。引擎用 `O_DIRECT` + 原生 AIO（`ple_ssd_io.so`）从这里直读，**不进显存** |
| `model-00002…00011-of-00011.safetensors` | 各 ≈5.0 GB | 主干权重（进显存，约 47 GiB） |
| `mtp-model-00001/00002-of-00002.safetensors` | 0.53 + 0.90 GB | MTP 投机解码头 |
| `model.safetensors.index.json` | 24 MB | `weight_map`：PLE 表按 key 定位到分片（缺它就启动不了） |

---

## 4. 部署步骤（8 步，每步：做什么 → 命令 → 期望）

> 每一步都可以用 `./deploy.sh check` 里的同名检查项独立验收。

### 4.0 先体检

```bash
git clone git@github.com:<你>/qwen3.8-flash-next-cmp170hx.git && cd qwen3.8-flash-next-cmp170hx
./deploy.sh check          # 只读，不改任何东西；有 FAIL 就按「→ 修复」逐条处理
./deploy.sh check --json   # 机器可读，便于 CI
```
**期望**：`host.*` 全 PASS（GPU/内存/磁盘/网络模式）。

### 4.1 配置：只要看一个文件

```bash
$EDITOR config/engine.env   # 端口、模型路径、上下文、MTP、PLE offload、治愈线程……
./start.sh --params         # 打印此刻生效的值与来源（不启动任何东西）
```
`config/engine.env` 是**全项目唯一默认值来源**（`bin/*`、`vllm-native/bin/run_native.sh`、`ops/bench/*.py` 都读它）。
**改端口/参数只改这一个文件**；临时覆盖用环境变量：`QWEN_MTP=1 ./start.sh`。

### 4.2 取模型权重（142.5 GiB，小时级）

```bash
# 本机用的多连接续传下载器（hf-mirror + 48 路 range 请求，状态写 <模型目录>/.fdl-state.json）
~/dlvenv/bin/python ops/tools/fdl.py        # 中断后重跑即续传，不必重头来
# 备选：镜像时代的官方脚本（容器里跑 snapshot_download，revision 已钉 ce0e0b94…）
#   scripts/download-model.sh
```
**验证**（必须全绿，否则引擎会加载失败或静默退化）：
```bash
python3 ops/tools/verify_ckpt.py            # 逐分片校验 safetensors 头 vs 文件长度
./deploy.sh check                           # model.dir / model.index / model.shards / model.ple
```
**期望**：`13 个分片 / 142.5GiB`，且最大分片 `model-00001-of-00011.safetensors = 102.4 GB`（PLE 表）。
revision 必须钉在 `ce0e0b94083895bd836b916f29bf105c40a8162a`（`scripts/common.sh` 里的 `MODEL_REVISION`）。

### 4.3 运行时（两条路线，二选一）

**路线 A —— 从 Docker 镜像 payload 提取（本机当前这份运行时就是这么来的；重跑未验证）**

镜像里已经有一份装好的运行时，`ops/OPS.md §9.22.2` 记录了当时的做法与校验：

1. 从镜像层拷出 `opt/vllm/{.venv,src,optimization}` + `opt/{entrypoint.sh,download-model.sh}` + `uv-python/cpython-3.12.14-linux-x86_64-gnu`，
   放到 `vllm-native/`（跳过 `/opt/nvidia`，1.3 GB 的 nsight，用不到）。
2. **路径改写**（漏一条就 `ModuleNotFoundError` 或 shebang 跑飞）：`.venv/bin/*` 里 61 个脚本的 shebang、
   `.venv/pyvenv.cfg` 的 `home`、`.venv/lib/python3.12/site-packages/__editable__*.pth`、
   `direct_url.json`、`uv-python/cpython-3.12-linux-x86_64-gnu` 符号链接、`.venv/bin/python` 链接。
3. **校验**：与镜像内文件数逐项比对（68860 / 3732 一致），且导入清单逐项 diff 为空
   （torch 2.13.0+cu130 / triton 3.7.1 / vllm …ple1 / flashinfer 0.6.18 / tokenspeed_triton 3.8.10 /
   `_C_stable_libtorch` / `_custom_ops` / `_flashmla_C` / `fs_io_C` / qsa）。

> 注意镜像已删除（铁律 11），要重走这条路得先 `docker pull 18gogogo/170hx1-qwen38nextf@sha256:9d8f3bab…`
> （凭据与校验见 `ops/legacy-docker/IMAGE-PROVENANCE.md`）。

**路线 B —— 用上游脚本自己构建（可复现；本机未走此路线，未验证）**

```bash
# ① 克隆 vLLM 到 $VLLM_WORKDIR/src（install-runtime.sh 的布局假设）
VLLM_WORKDIR="$PWD/vllm-native/opt/vllm"
git clone https://github.com/vllm-project/vllm "$VLLM_WORKDIR/src"
git -C "$VLLM_WORKDIR/src" checkout a5a30471ff2bb7f0824f2da10e358af98d304472   # scripts/common.sh 里的 VLLM_REVISION
# ② 装 venv + 预编译轮子 + editable 安装 + 编 ple_ssd_io.so（脚本自己串联这几步）
VLLM_WORKDIR="$VLLM_WORKDIR" scripts/install-runtime.sh
```
细节（`scripts/install-runtime.sh` 干的事）：`uv venv --python 3.12` → 装 wheel
`wheels.vllm.ai/$VLLM_REVISION/vllm-0.29.1rc1.dev402+ga5a30471f-cp38-abi3-manylinux_2_28_x86_64.whl`
→ 装 `requirements/build/cuda.txt` + `huggingface-hub` → 用 `VLLM_USE_PRECOMPILED=1` /
`VLLM_VERSION_OVERRIDE=0.29.1rc1.dev402+ga5a30471f.ple1` / `--config-settings editable_mode=compat -e .`
做 editable 安装（**editable 很关键**：补丁改的是 `src/`，改完立即生效，不必重装）。
⚠️ 脚本里 `VLLM_WORKDIR` 默认是 **`$HOME/vllm`（已废弃路径，铁律 10）**，所以务必显式传入（如上）。
⚠️ 还需要 `uv`，以及 `uv-python`（uv 自带的 CPython 3.12）——它会是 `.venv/bin/python` 的软链目标。

### 4.4 打补丁（两个，缺一不可）

```bash
SRC=vllm-native/opt/vllm/src
git -C $SRC rev-parse HEAD      # 必须是 a5a30471ff2bb7f0824f2da10e358af98d304472

# ① 上游的 PLE SSD offload 补丁（1345 行，命中 11 个文件 + 新增 ple_ssd.py / ple_ssd_io.c）
VLLM_WORKDIR="$PWD/vllm-native/opt/vllm" scripts/apply-patch.sh
# ② 本地分配器治愈线程（65 行，命中 ops/qsa.py）——apply-patch.sh 不管这个，手工打
git -C $SRC apply --check ops/patches/qsa-alloc-heal.patch && git -C $SRC apply ops/patches/qsa-alloc-heal.patch
```
**验证**（反向校验 = "补丁的产物已经在树里"，这是最便宜的判据）：
```bash
for p in patches/qwen38-ple-ssd.patch ops/patches/qsa-alloc-heal.patch; do
    git -C vllm-native/opt/vllm/src apply --check --reverse "$p" && echo "OK $p"
done
./deploy.sh check     # patch.rev / patch.* / patch.all
```
> ⚠️ **引擎在跑的时候不要打补丁**（`deploy.sh install --yes` 会主动拒绝）。先 `bin/stop.sh`，并且要征得用户同意。
>
> 已知差异：本机 `src/.../ops/qsa.py` 的工作区比 `qsa-alloc-heal.patch` 多出 14+/2- 行 —— 是已退役的
> `cg_instr.stage("ATTENTION")` 插桩残留（模块不存在 ⇒ 整段是 no-op，`ops/OPS.md:450/538/551`）。
> 按补丁重建得到的树**不含**这段死代码，是更干净的状态。

### 4.5 编 `ple_ssd_io.so`（秒级）

```bash
scripts/build-ple-io.sh     # cc -O3 -shared -fPIC -Wall -Wextra -Werror
                            # 源文件: $SRC/vllm/models/qwen4_exp/nvidia/ple_ssd_io.c
                            # 产物:   $ROOT/vllm-native/opt/vllm/optimization/ple_ssd_io.so（15 832 字节）
```
**验证**：`./deploy.sh check` 的 `ple.lib` 会 `ctypes.CDLL` 真的 dlopen 一次（不碰 GPU）。
路径由 `config/engine.env` 的 `QWEN_PLE_LIB` 注入 `--additional-config ple_ssd_native_library`。

### 4.6 Triton 缓存权限（一个小坑能直接拦住启动）

```bash
ls -ld vllm-native/triton_cache        # 必须是当前用户可写，属主 hong
```
Docker 时代缓存是 **root** 建的，直接挪过来会让 `run_native.sh start` **直接报错退出**。修复：
```bash
sudo chown -R "$(id -un):$(id -gn)" vllm-native/triton_cache          # 二是改属主
cp -r ops/legacy-docker/triton_cache/. vllm-native/triton_cache/      # 或从旧缓存拷一份（省一次冷启动编译）
```

### 4.7 启动

```bash
./deploy.sh check          # 先确认没有 FAIL
./start.sh                 # 启动 → 等 health=200（约 250~320 s）→ 自动回收 Windows 页缓存
#   --wait 900             就绪等待上限（默认 600 s，见 QWEN_START_WAIT）
#   --keep-cache           不回收宿主页缓存（不推荐：会让整机发卡、解码变慢）
#   --foreground           前台跑（调试用，Ctrl-C 退出）
```
调用链（不要绕过它）：
```
./start.sh → bin/start.sh → vllm-native/bin/run_native.sh start → setsid nohup vllm serve …（脱离会话）
```
**期望**：结尾打印 `接口 http://127.0.0.1:8000/v1`、`pid …`、`显存 …`。
引擎日志在 `vllm-native/logs/server.log`（>100 MB 自动轮转，留 5 份）。

### 4.8 验收

```bash
./deploy.sh verify            # health + 新鲜 2048 预填 ×3（判读见 §5）
./deploy.sh verify --full     # 追加 bin/bench.sh --full：131072 长上下文 + 事后中毒检查（约 3 分钟）
bin/status.sh --watch 5       # 实时看：请求数 / KV / MTP 接受率 / 宿主内存
```

---

## 5. 验收基线与判读

**基线（`docs/RESULTS-WSL2.md`，本机 native pass3 + 2026-10-01 复测；偏离超过 ~1.5× 就该查）**

| 指标 | 基线 | 判据 |
|---|---|---|
| `/health` | 200 | 不是 200 ⇒ 看 `bin/logs.sh -e` |
| 预填 2048（新鲜 id） | 0.64~0.73 s（中位 0.693） | ≤0.80 PASS；≤1.15 WARN；>1.15 FAIL |
| 预填 8192 | 2.0~2.4 s（中位 2.184） | 同上比例 |
| 预填 131072 | **48.7~49.0 s（2 694 tok/s）**（2026-10-01 复测）；9 月旧记录 54.6 s / 2 399 tok/s | `verify --full` 会跑 |
| 解码步时 | 15.2~15.7 ms（MTP=1）/ 17.2~17.6 ms（MTP=2） | 需先预热 ≥2500 token 且 `vmmemWSL < 32 GB`（铁律 3） |
| 稳态解码 | ≈111 tok/s（MTP=1）/ ≈126 tok/s（MTP=2） | `bin/bench.sh` |
| MTP 接受率 | ≈72% | `bin/status.sh` |
| 长上下文后"中毒"检查 | 后置全新 2048 预填 ≈0.68 s | 变慢 ⇒ 治愈线程没生效（铁律 5） |

**`./deploy.sh verify` 的三种结论**：`PASS` = 命中基线；`WARN` = 明显偏慢但仍可用（先 `bin/drop_host_cache.sh`，
再确认 `QSA_ALLOC_HEAL=1`，再看是否只是宿主/磁盘在忙）；`FAIL` = 约 2× 以上，按提示逐条排查。

**测量纪律**（否则数字全是假的）：每次换新鲜的 token id（`ops/bench/warmup.py` 已经这么做）——
固定 seed 的 prompt 会命中前缀缓存（32K 预填 10.4 s → 0.68 s，铁律 4）。

---

## 6. 日常运维

| 想干什么 | 命令 |
|---|---|
| 启动 / 停止 / 重启 | `./start.sh`、`bin/stop.sh`（先 `--check`）、`bin/stop.sh && ./start.sh` |
| 看状态（单行 / 实时） | `bin/status.sh --short`、`bin/status.sh --watch 5` |
| 看日志 | `bin/logs.sh -f`、`-e`（错误）、`--startup`、`--heal`（治愈线程）、`--list`（归档） |
| 体检 + 存档 | `bin/bench.sh`（追加 `ops/measurements/perf-history.csv`）、`--full`、`--tag mtp3` |
| 还内存给 Windows | `bin/drop_host_cache.sh`（加载模型后 / 解码测速前必跑） |
| 只读看板 | `ops/tools/metrics_web.sh start` → `http://127.0.0.1:9494` |
| 客户端 | `http://127.0.0.1:8000/v1`，模型名 `Qwen3.8-Flash-Next`；**Windows 侧不能用 `localhost`**（铁律 2） |

---

## 7. 故障排查（现象 → 根因 → 处置）

| 现象 | 根因 | 处置 | 出处 |
|---|---|---|---|
| `run_native.sh start` 立刻报 `TRITON_CACHE_DIR 不可写` | 缓存是 root 建的（Docker 时代） | `sudo chown -R $(id -un) vllm-native/triton_cache` | §4.6 |
| 启动后 `health=000` 但进程在 | 还在加载 143 GB 权重 | 等满 250~320 s；`bin/logs.sh --startup` 看进度 | §4.7 |
| 端到端超时 / 连不上 | Windows 侧用了 `localhost`（解析成 `::1`，mirrored 不转发） | 改用 `127.0.0.1` | 铁律 2 |
| 解码比基线慢 25~40% | 少预热（<2500 token）或宿主内存高（`vmmemWSL ≥ 32 GB`） | 先预热再测；`bin/drop_host_cache.sh` | 铁律 3 |
| 预填 2048 突然 2 s 以上 | 分配器"中毒"（≥96K 预填后 device free=0 + 碎片） | 确认 `QSA_ALLOC_HEAL=1`（`bin/logs.sh --heal` 看 `QXHEAL fired`）；严重时重启 | 铁律 5 |
| 长上下文 prefill 慢 4.5×、解码慢 1.7× | 手滑关掉了治愈线程 | 把 `QWEN_ALLOC_HEAL=1` 打开并**重启** | 铁律 5 |
| 整机发卡、鼠标卡顿 | WSL 攒了 ~50 GB 干净页缓存 | `bin/drop_host_cache.sh` | 铁律 7 |
| 一开长上下文/跑探针就蓝屏或整机冻 | 引擎内 CUDA 探针/电池 → 主机 `dxgkrnl` WATCHDOG 活转储 | **永久退役**，别跑 `*_battery`/QXPROBE/QXGATE | 铁律 6 |
| NVMe 掉盘 → WSL/引擎一起崩（`0x7A`） | D: KIOXIA 掉盘；引擎在跑时做 GB 级删除/全盘扫描会诱发 | 停服务后再清理；关 ASPM/PCIe 链路电源管理 | 铁律 12 |
| `bin/stop.sh` 提示 60 s 未退出 | 还有请求在跑 | 先 `bin/logs.sh -n 80`；确认空闲后才 `QWEN_FORCE=1 bin/stop.sh` | 铁律 8 |
| 权重目录少分片 / 大小不对 | 下载中断 | 重跑 `ops/tools/fdl.py`（`.fdl-state.json` 续传）+ `ops/tools/verify_ckpt.py` | §4.2 |
| 端口冲突 / 改端口不生效 | 只改了一个地方（历史上踩过：8001/9393 混用） | 只改 `config/engine.env`，然后 `bin/stop.sh && ./start.sh`；`deploy.sh check` 的 `config.port` 会比对运行中进程 | `docs/SCRIPTS.md §5` |

---

## 8. 回退到 Docker（应急路径）

镜像**本机已删除**（2026-09-29，释放 25 GB；铁律 11），回退前必须：
```bash
docker pull 18gogogo/170hx1-qwen38nextf@sha256:9d8f3bab…    # digest 与校验见 ops/legacy-docker/IMAGE-PROVENANCE.md
bin/stop.sh --check && bin/stop.sh
ops/legacy-docker/run_container.sh
```
⚠️ 回退路径与原生路径**不共享配置**（容器里的 `/opt/entrypoint.sh` 是另一套默认值，端口 9393/8000 历史都见过）：
不要混用 `ops/diagnostics/*`（硬编码 `localhost:9393`）与原生脚本。Docker 路径的原始参数在 `vllm-native/opt/entrypoint.sh`（已入库，可对照）。

---

## 9. 铁律速查（全文见 `AGENTS.md` 第 3 节）

1. 每次改动都要以"可用且已验证"结束：`bin/status.sh` → `health=200` + 跑一次真实预填。
2. **别用 `localhost`**，Windows 侧必须 `127.0.0.1`。
3. 测解码前预热 ≥2500 token 且 `vmmemWSL < 32 GB`，先 `bin/drop_host_cache.sh`。
4. 预填测量每次换 seed / 用新鲜 token id（否则命中前缀缓存）。
5. ≥96K 预填后检查"中毒"（后置全新 2048 应 ≈0.68 s）；`QSA_ALLOC_HEAL=1` 不要关。
6. 引擎内 CUDA 探针永久退役（会让整机冻结 + 写 1~5 GB 崩溃转储）。
7. 每次加载模型后跑 `bin/drop_host_cache.sh`。
8. 停服务用 `bin/stop.sh`（先 `--check`），不要 `kill -9`。
9. 重启 = 4~5 分钟不可用；`.wslconfig` 改动要 `wsl --shutdown` —— 两者都要先问用户。
10. `~/vllm` 已废弃，工具都在 `ops/`。
11. Docker 镜像已删，回退要先按 digest 重新 pull。
12. NVMe 会掉盘：引擎在跑时不要做 GB 级删除或全盘扫描。

---

## 10. 已知文档陈旧点（诚实标注，未改）

* `docs/RESULTS-WSL2.md` 的 *Reproducing* 段仍写 `--url http://127.0.0.1:9393`，正文也仍说"`:8000` 转发到 `:9393`" ——
  2026-10-01 起端口已统一为 **8000**、转发层已删（`ops/OPS.md §9.32`）。按**本文件**的命令跑。
* `ops/tools/fdl_supervisor.sh` 里引用的是 `$HOME/vllm/fdl.py` 与 `$HOME/dlvenv`（旧路径）；直接用
  `ops/tools/fdl.py` 即可，或把该脚本的路径改成仓库内路径。
* `docs/SCRIPTS.md §3.3` 提到 `ops/tools/metrics_web.sh` 默认 `127.0.0.1:9494`（正确），但其上游端口取
  `QWEN_PORT`（8000）——两者不要混。
* `ops/OPS.md:1478` 提到的 `vllm-native/bin/drop_host_cache.sh` 已改名到 `bin/drop_host_cache.sh`。
