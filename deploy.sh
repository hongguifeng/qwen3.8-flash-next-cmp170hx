#!/usr/bin/env bash
# WSL2 部署入口（薄封装，完全等价于 ops/deploy/deploy_wsl2.sh）。
#
# 用法:
#   ./deploy.sh check            # 【默认】只读体检：环境/GPU/内存/磁盘/模型/运行时/补丁/服务
#   ./deploy.sh check --json     # 机器可读输出（CI 用）
#   ./deploy.sh plan             # 打印 8 步部署流程（不动任何东西）
#   ./deploy.sh install          # 报告缺什么（不动手）；install --yes 幂等补齐
#   ./deploy.sh start            # 拉起服务（委托 bin/start.sh / run_native.sh）
#   ./deploy.sh verify           # 验收：health=200 + 新鲜 2048 预填（对比基线）
#   ./deploy.sh all              # check → start → verify
#   ./deploy.sh --help           # 看 ops/deploy/deploy_wsl2.sh 的完整说明
#
# 完整部署文档: docs/DEPLOY-WSL2.md
# 安全默认：只读。唯一会改磁盘的入口是 `install --yes`（且引擎在跑时拒绝打补丁）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
    -h|--help) exec "$ROOT/ops/deploy/deploy_wsl2.sh" --help ;;
esac

exec "$ROOT/ops/deploy/deploy_wsl2.sh" "$@"
