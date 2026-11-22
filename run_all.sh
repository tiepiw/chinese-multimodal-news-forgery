#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for MODEL in textcnn bert concat eann cafe hammer arg himf_moe qwen35_4b; do
  bash "$ROOT/scripts/run_${MODEL}.sh" "$@"
done
