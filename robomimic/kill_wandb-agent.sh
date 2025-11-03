#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date +'%F %T')] $*"; }

# 子プロセスを再帰的に列挙
children_of() {
  local p="$1"
  ps -eo pid=,ppid= | awk -v PPID="$p" '$2==PPID {print $1}'
}

kill_tree() {
  local root="$1" timeout="${2:-5}"
  # 先に子孫を全て止める
  for c in $(children_of "$root"); do
    kill_tree "$c" "$timeout"
  done
  if kill -0 "$root" 2>/dev/null; then
    kill -TERM "$root" 2>/dev/null || true
  fi
}

force_kill_tree() {
  local root="$1"
  for c in $(children_of "$root"); do
    force_kill_tree "$c"
  done
  if kill -0 "$root" 2>/dev/null; then
    kill -KILL "$root" 2>/dev/null || true
  fi
}

log "WandB関連プロセス一覧"
ps -eo pid,ppid,state,etime,cmd --sort=ppid,pid | awk 'NR==1 || tolower($0) ~ /wandb/ {print}'

echo
log "ゾンビプロセス (STATE=Z) の一覧"
ps -eo pid,ppid,state,etime,comm --no-headers | awk '$3=="Z"{printf "ZOMBIE  PID=%s  PPID=%s  ELAPSED=%s  COMM=%s\n",$1,$2,$4,$5}'

echo
log "wandb agent / 関連プロセスを検出"
# agent 本体（様々な起動形態をカバー）
mapfile -t AGENTS < <(ps -eo pid,comm,args --no-headers | \
  awk '/[Ww]andb([ -]|-)?agent/ || /python[0-9.]* .*wandb[^ ]* agent/ {print $1}' | sort -u)

# サービス系（必要に応じて落とす）
mapfile -t SERVICES < <(ps -eo pid,comm,args --no-headers | \
  awk '/wandb[- ](service|local|sync|launcher)/ {print $1}' | sort -u)

TARGETS=("${AGENTS[@]}" "${SERVICES[@]}")

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  log "対象プロセスは見つかりませんでした"
else
  echo "対象PIDs: ${TARGETS[*]}"

  log "SIGTERM を送信（プロセスツリーごと）"
  for pid in "${TARGETS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill_tree "$pid" 5
    fi
  done

  # 待機
  sleep 3

  # まだ生存しているものは SIGKILL
  STILL=()
  for pid in "${TARGETS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      STILL+=("$pid")
    fi
  done

  if [[ ${#STILL[@]} -gt 0 ]]; then
    log "まだ生存中のため SIGKILL を送信（プロセスツリーごと）: ${STILL[*]}"
    for pid in "${STILL[@]}"; do
      force_kill_tree "$pid"
    done
  fi
fi

echo
log "ゾンビの親に SIGCHLD / HUP を送って回収を促進"
ppids=$(ps -eo pid,ppid,state,comm --no-headers | awk '$3=="Z"{print $2}' | sort -u)
if [[ -n "${ppids}" ]]; then
  for p in $ppids; do
    if kill -0 "$p" 2>/dev/null; then
      kill -CHLD "$p" 2>/dev/null || true
      kill -HUP  "$p" 2>/dev/null || true
      echo " -> signaled PPID=$p"
    fi
  done
else
  echo " (ゾンビの親は見つからず)"
fi

echo
log "後確認（残存する wandb 関連 / ゾンビ）"
ps -eo pid,ppid,state,etime,cmd --sort=ppid,pid | awk 'NR==1 || tolower($0) ~ /wandb/ {print}'
ps -eo pid,ppid,state,etime,comm --no-headers | awk '$3=="Z"{printf "ZOMBIE  PID=%s  PPID=%s  ELAPSED=%s  COMM=%s\n",$1,$2,$4,$5}'

echo
log "完了"