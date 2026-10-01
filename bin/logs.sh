#!/usr/bin/env bash
# 查看推理服务日志。
#
# 用法:
#   bin/logs.sh                  # 末尾 80 行
#   bin/logs.sh -f               # 持续跟踪（Ctrl-C 退出）
#   bin/logs.sh -n 300           # 末尾 300 行
#   bin/logs.sh -e               # 只看错误/异常/回溯
#   bin/logs.sh --heal           # 只看内存治愈线程（QXHEAL）的动作
#   bin/logs.sh --startup        # 只看启动关键行（模型加载/显存/CUDA graph/监听端口）
#   bin/logs.sh --slow           # 只看 prefill/decode 相关的耗时警告
#   bin/logs.sh --list           # 列出所有日志文件（含轮转归档）与大小
#   bin/logs.sh --path           # 只打印当前日志路径（配合 watch/tail 用）
#   bin/logs.sh --clean          # 归档当前日志（server-<时间>.log）再清空；引擎不用停
#   bin/logs.sh --help
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

N=80; FOLLOW=0; MODE=plain
while [ $# -gt 0 ]; do
    case "$1" in
        -f|--follow) FOLLOW=1 ;;
        -n) N="${2:-80}"; shift ;;
        -e|--errors) MODE=errors ;;
        --heal) MODE=heal ;;
        --startup) MODE=startup ;;
        --slow) MODE=slow ;;
        --list) MODE=list ;;
        --path) echo "$LOG_FILE"; exit 0 ;;
        --clean) MODE=clean ;;
        -h|--help) help_exit "$0" ;;
        *) echo "未知参数: $1（--help 看用法）" >&2; exit 2 ;;
    esac
    shift
done

[ -f "$LOG_FILE" ] || { echo "日志文件不存在: $LOG_FILE（服务可能从未启动过）" >&2; exit 1; }

case "$MODE" in
    list)
        hr; printf '日志目录 %s\n' "$LOG_DIR"; hr
        ls -lhtr "$LOG_DIR" | tail -n +2 | awk '{s=$5; if (s<1024) u=s" B"; else if (s<1048576) u=sprintf("%.1f KB",s/1024); else u=sprintf("%.1f MB",s/1048576); printf "  %-34s %10s  %s %s\n", $9, u, $6, $7}'
        echo; echo "当前日志: $LOG_FILE"
        ;;
    clean)
        # 安全前提：写入者必须以 O_APPEND 打开日志（logrotate 的 copytruncate 语义）。
        # 若引擎以普通重定向（无 O_APPEND）持有 fd，它就记得自己的写入偏移量，
        # 截断后下一次写入会落在旧偏移上 ⇒ 文件头出现一大段 NUL 空洞。那种情况下只归档、不截断。
        pid="$(engine_pid)"
        if [ ! -s "$LOG_FILE" ]; then
            echo "当前日志是空的，无需归档（$LOG_FILE）"
            exit 0
        fi
        append_safe=1; why=""
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            flags="$(awk '/^flags:/{print $2}' "/proc/$pid/fdinfo/1" 2>/dev/null)"
            if [ -n "$flags" ] && (( 8#$flags & 8#2000 )); then
                append_safe=1
            else
                append_safe=0
                why="引擎 pid $pid 的 stdout 不是 O_APPEND（fdinfo/1 flags=${flags:-读不到}）"
            fi
        fi
        archive="$LOG_DIR/server-$(date +%Y%m%d-%H%M%S).log"
        cp "$LOG_FILE" "$archive"
        arch_size="$(du -h --apparent-size "$archive" | cut -f1)"
        if [ "$append_safe" = 1 ]; then
            : > "$LOG_FILE"
            echo "已归档 → $archive （$arch_size）"
            echo "当前日志已清空，引擎继续正常写：$LOG_FILE"
        else
            echo "已归档 → $archive （$arch_size）"
            c_warn "但【没有清空】当前日志"
            printf '  原因：%s\n' "$why"
            echo "  截断会在文件头留下与写入偏移等长的 NUL 空洞（引擎不重启就没法重置偏移）。"
            echo "  要真正清空：bin/stop.sh → bin/logs.sh --clean → bin/start.sh；"
            echo "  或者之后重启一次（已改为 O_APPEND 写入，那时 --clean 就能安全截断）。"
            exit 0
        fi
        ;;
    errors)
        hr; echo "错误/异常（最近 $N 条匹配）"; hr
        grep -aiE 'traceback|error|exception|failed|assert|illegal|CUDA error|out of memory' "$LOG_FILE" | tail -n "$N"
        echo; echo "提示：完整上下文用 bin/logs.sh -n 400"
        ;;
    heal)
        hr; echo "内存治愈线程 QXHEAL 动作"; hr
        grep -a 'QXHEAL' "$LOG_FILE" | tail -n "$N"
        printf '\n实际回收动作触发次数: %s\n' "$(grep -ac 'QXHEAL fired' "$LOG_FILE")"
        ;;
    startup)
        hr; echo "启动关键行"; hr
        grep -aE 'Starting vLLM|Loading model|Loading weights|weights took|Capturing|Captur|CUDA graph|Started server|Uvicorn running|Available|listening on|init engine|memory profiling' "$LOG_FILE" | tail -n "$N"
        ;;
    slow)
        hr; echo "耗时/性能相关行"; hr
        grep -aiE 'slow|took [0-9]|tok/s|prefill|chunk|throttl|cooldown' "$LOG_FILE" | tail -n "$N"
        ;;
    *)
        if [ "$FOLLOW" = 1 ]; then
            hr; echo "跟踪 $LOG_FILE （Ctrl-C 退出）"; hr
            tail -n "$N" -f "$LOG_FILE"
        else
            hr; echo "末尾 $N 行（共 $(wc -l < "$LOG_FILE") 行）: $LOG_FILE"; hr
            tail -n "$N" "$LOG_FILE"
        fi
        ;;
esac
