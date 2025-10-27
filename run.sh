#!/usr/bin/env bash
set -euo pipefail

# Local launcher for the ultrasonic visualizer in this repo.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
ENTRY="$ROOT_DIR/ultrasonic_visualizer.py"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Creating virtualenv and installing dependencies..."
  python3 -m venv "$ROOT_DIR/.venv"
  source "$ROOT_DIR/.venv/bin/activate"
  pip install -U pip
  pip install -e "$ROOT_DIR"
else
  source "$ROOT_DIR/.venv/bin/activate"
fi

# Prefer an interactive Matplotlib backend when a display is available.
if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
  if python -c "import tkinter" >/dev/null 2>&1; then
    export MPLBACKEND=TkAgg
  fi
fi

exec python "$ENTRY" "$@"

