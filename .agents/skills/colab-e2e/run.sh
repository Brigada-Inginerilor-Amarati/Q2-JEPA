#!/usr/bin/env bash
# End-to-end GPU check on Colab: pushed main -> T4 session -> notebooks/colab_gpu.ipynb -> verify -> stop.
# Usage: run.sh [--drive] [-s NAME] [--gpu T4|L4|A100] [--timeout SECONDS]
#   --drive  pause so a human can run `colab drivemount` (checkpoints persist); default is the
#            scratch fallback dir, whose checkpoints are lost at `colab stop`.
set -euo pipefail

SESSION=q2jepa-e2e GPU=T4 TIMEOUT=600 DRIVE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --drive) DRIVE=1 ;;
    -s) SESSION=$2; shift ;;
    --gpu) GPU=$2; shift ;;
    --timeout) TIMEOUT=$2; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
die() { echo "colab-e2e: $*" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"

# The notebook clones origin's default branch, so only pushed commits on main get tested.
[ -z "$(git status --porcelain --untracked-files=no)" ] || die "uncommitted changes would not reach Colab; commit + push first"
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || die "HEAD is not origin/main; push first (or edit the notebook clone cell for a branch)"

colab whoami >/dev/null || die "colab auth failed; see .agents/skills/colab-operator/SKILL.md (Authentication)"
colab new -s "$SESSION" --gpu "$GPU"
trap 'colab stop -s "$SESSION"' EXIT

if [ "$DRIVE" = 1 ]; then
  echo "In another terminal run:  colab drivemount -s $SESSION   then press Enter here."
  read -r
else
  echo "colab-e2e: scratch mode, checkpoints are NOT persisted"
  echo 'import os; os.makedirs("/content/drive/MyDrive", exist_ok=True)' | colab exec -s "$SESSION"
fi

colab exec -s "$SESSION" --timeout "$TIMEOUT" -f notebooks/colab_gpu.ipynb

# `!cmd` failures don't raise in IPython, so scan the output text, not just error outputs.
python3 - notebooks/colab_gpu_output.ipynb <<'EOF'
import json, re, sys
cells = json.load(open(sys.argv[1]))["cells"]
outs = [o for c in cells if c["cell_type"] == "code" for o in c.get("outputs", [])]
text = "".join("".join(o.get("text", "")) + "".join(o.get("traceback", [])) for o in outs)
bad = [o for o in outs if o.get("output_type") == "error"] or re.findall(r"\b[1-9]\d* (?:failed|errors?)\b|Traceback", text)
if bad or "passed" not in text or "'coverage'" not in text:
    sys.exit(f"colab-e2e: FAILED, read notebooks/colab_gpu_output.ipynb ({bad[:3] if bad else 'missing tests or metrics'})")
print("colab-e2e: OK (tests passed, metrics printed)")
EOF
