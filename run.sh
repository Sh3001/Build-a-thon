#!/usr/bin/env bash
# RecoverAI launcher. Resolves its own directory and its own interpreter, so it works
# from anywhere and cannot pick up the sibling Vasooli project's venv or scripts.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"
cd "$HERE"

if [[ ! -x "$PY" ]]; then
  echo "error: $PY not found." >&2
  echo "  create it with:  python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e ." >&2
  exit 1
fi

usage() {
  cat <<'USAGE'
RecoverAI

  ./run.sh serve [PORT]   serve the dashboard + API   (default 8000)
  ./run.sh demo [ARGS]    full pipeline, then serve   (--quick, --no-serve, ...)
  ./run.sh test [ARGS]    run the test suite
  ./run.sh pipeline       dataset -> train -> evaluate -> experiment (no server)
  ./run.sh shell          a python REPL with the project importable
USAGE
}

cmd="${1:-}"; shift || true
case "$cmd" in
  serve)
    port="${1:-8000}"
    echo "→ http://127.0.0.1:$port/"
    exec "$PY" -m uvicorn backend.app.main:app --port "$port"
    ;;
  demo)     exec "$PY" scripts/demo.py "$@" ;;
  test)     exec "$PY" -m pytest backend/tests/ "${@:--q}" ;;
  pipeline)
    "$PY" scripts/generate_dataset.py
    "$PY" ml/train.py
    "$PY" ml/evaluate.py
    exec "$PY" scripts/run_experiment.py --fresh
    ;;
  shell)    exec "$PY" ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown command: $cmd" >&2; echo; usage; exit 2 ;;
esac
