#!/usr/bin/env bash
set -euo pipefail

# Universal launcher for the ultrasonic visualizer.
# Works whether the code lives in 'ultrasonic-visualizer/' (new repo)
# or the legacy 'usdetect/' tree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional override: USDETECT_PROJECT_DIR or --project-dir/-p
PROJECT_DIR="${USDETECT_PROJECT_DIR:-}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project-dir)
      [[ $# -ge 2 ]] || { echo "--project-dir requires a path" >&2; exit 2; }
      PROJECT_DIR="$2"; shift 2 ;;
    --) shift; while [[ $# -gt 0 ]]; do ARGS+=("$1"); shift; done ; break ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

discover_project_dir() {
  local candidates=()
  candidates+=("$SCRIPT_DIR" "$PWD" "$(dirname "$SCRIPT_DIR")" "$(dirname "$PWD")")
  for base in "${candidates[@]}"; do
    [[ -d "$base" ]] || continue
    for d in "$base"/*; do [[ -d "$d" ]] && candidates+=("$d"); done
  done
  # Deduplicate
  local seen tmp=()
  for c in "${candidates[@]}"; do
    [[ -n "${seen:-}" ]] && [[ ",${seen}," == *",${c},"* ]] && continue
    tmp+=("$c"); seen+="${c},"
  done
  candidates=("${tmp[@]}")
  for c in "${candidates[@]}"; do
    if [[ -x "$c/ultrasonic-visualizer/run.sh" ]] || [[ -f "$c/ultrasonic-visualizer/ultrasonic_visualizer.py" ]] || [[ -f "$c/usdetect/ultrasonic_visualizer.py" ]]; then
      echo "$c"; return 0
    fi
  done
  return 1
}

if [[ -z "$PROJECT_DIR" ]]; then
  PROJECT_DIR="$(discover_project_dir)" || {
    echo "Error: could not locate project directory containing ultrasonic-visualizer or usdetect." >&2
    echo "Hint: set USDETECT_PROJECT_DIR or pass --project-dir /path/to/repo" >&2
    exit 1
  }
fi

# Prefer new standalone repo if present
if [[ -x "$PROJECT_DIR/ultrasonic-visualizer/run.sh" ]]; then
  exec "$PROJECT_DIR/ultrasonic-visualizer/run.sh" "${ARGS[@]}"
fi

# Legacy path: vc01/usdetect
VENV_PY="$PROJECT_DIR/usdetect/.venv/bin/python"
VISUALIZER="$PROJECT_DIR/usdetect/ultrasonic_visualizer.py"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: venv Python not found at $VENV_PY" >&2
  echo "Create it and install deps, e.g.:" >&2
  echo "  cd $PROJECT_DIR/usdetect && python3 -m venv .venv && source .venv/bin/activate" >&2
  echo "  pip install numpy sounddevice matplotlib" >&2
  exit 1
fi
[[ -f "$VISUALIZER" ]] || { echo "Error: $VISUALIZER not found" >&2; exit 1; }

# GUI backend hint
if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
  if "$VENV_PY" -c "import tkinter" >/dev/null 2>&1; then
    export MPLBACKEND=TkAgg
  fi
fi

exec "$VENV_PY" "$VISUALIZER" "${ARGS[@]}"

