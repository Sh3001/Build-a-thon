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
  ./run.sh multiseed [N]  seven arms across N seeds with confidence intervals (default 20)
  ./run.sh sweep [N]      the same, across all nine simulator scenarios (default 8)
  ./run.sh verify [CASE]  verify the audit chain; with a transaction id, its timeline
  ./run.sh bench          throughput and per-case latency at increasing batch sizes
  ./run.sh check          ruff + mypy on the safety-critical modules + the test suite
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
  multiseed) exec "$PY" scripts/run_multiseed.py --seeds "${1:-20}" "${@:2}" ;;
  sweep)     exec "$PY" scripts/run_multiseed.py --sweep --seeds "${1:-8}" "${@:2}" ;;
  verify)
    if [[ -n "${1:-}" ]]; then exec "$PY" scripts/verify_audit.py --case "$1"; fi
    exec "$PY" scripts/verify_audit.py
    ;;
  bench)    exec "$PY" scripts/bench_throughput.py "$@" ;;
  check)
    # The three gates CI runs, in the order that fails fastest.
    "$HERE/.venv/bin/ruff" check backend simulation scripts ml
    "$HERE/.venv/bin/mypy" backend/app/policies backend/app/domain \
        backend/app/security backend/app/decision
    exec "$PY" -m pytest backend/tests/ -q
    ;;
  shell)    exec "$PY" ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown command: $cmd" >&2; echo; usage; exit 2 ;;
esac
