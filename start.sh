#!/usr/bin/env bash
# 项目根目录的启动入口（薄封装，完全等价于 bin/start.sh——参数、默认值、行为都在那边）。
#
# 用法（常用场景两个位置参数就够了：档位 + 显卡）:
#   ./start.sh                 # 默认档位 main → GPU0 / :8000（等就绪 250~320 s，然后回收宿主内存）
#   ./start.sh unc             # 起未审查 checkpoint（GPU1 / :8001，可与 main 并行）
#   ./start.sh unc 0           # 起 unc 但放到 GPU0
#   ./start.sh 1               # 只换卡：main → GPU1
#   ./start.sh list            # 列出所有档位（模型目录 / 端口 / 显卡），不启动
#   ./start.sh --wait 900      # 就绪等待上限（秒）
#   ./start.sh --keep-cache    # 不回收 WSL 页缓存（默认会回收，防整机发卡）
#   ./start.sh --foreground    # 前台运行（调试用，Ctrl-C 退出）
#   ./start.sh --params        # 打印生效参数与来源，不启动
#   ./start.sh --help          # 看 bin/start.sh 的完整说明
#
# 管起来（带同一个档位名即可）:
#   bin/status.sh --model unc      bin/stop.sh --model unc      bin/logs.sh -f --model unc
#   bin/bench.sh --model unc --full
#
# 端口与全部引擎参数（含档位 = 模型 + 端口 + 显卡）的默认值只在 config/engine.env 一处定义：
#   QWEN_PORT=8000 QWEN_MTP=2 QWEN_BATCH_TOKENS=2048 ./start.sh     # 临时覆盖，不改文件
#
# 引擎由 vllm-native/bin/run_native.sh 用 `setsid nohup` 分离，所以本脚本退出后引擎继续跑。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
    -h|--help) exec "$ROOT/bin/start.sh" --help ;;
esac

exec "$ROOT/bin/start.sh" "$@"
