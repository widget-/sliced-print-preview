#!/usr/bin/env bash
# ────────────────────────────────────────────────────
# Pull latest code, rebuild frontend, restart backend.
# Run from your local machine — ssh-es into the VM.
# ────────────────────────────────────────────────────
set -euo pipefail

REMOTE="${1:-o}"  # default SSH alias

echo "→ Pulling latest code on ${REMOTE}..."
ssh "$REMOTE" "
  set -e
  cd ~/sliced-print-preview
  git pull

  echo '→ Installing dependencies...'
  cd packages/frontend
  ~/.bun/bin/bun install --frozen-lockfile 2>/dev/null || ~/.bun/bin/bun install

  echo '→ Building frontend...'
  ~/.bun/bin/bun run build

  echo '→ Restarting backend...'
  sudo systemctl daemon-reload
  sudo systemctl restart sliced-print-preview

  echo '✅ Deploy complete — backend restarted'
"
