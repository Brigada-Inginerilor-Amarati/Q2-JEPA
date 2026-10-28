---
name: colab-e2e
description: Run the full Q2-JEPA GPU check on a Colab T4 (quantum tests on nvidia, attention benchmark, training run) via the colab CLI, verify the output notebook, and always stop the VM.
disable-model-invocation: true
---

# Colab end-to-end run

Rents a billable GPU VM, so it only runs when the user asks. Background on the `colab` CLI (auth scopes,
recovery): [../colab-operator/SKILL.md](../colab-operator/SKILL.md). Workflow rules: `AGENTS.md` §5.

1. Make sure the work under test is committed and pushed to `origin/main`. The notebook clones GitHub,
   not the working tree; the script refuses to start otherwise. Ask before pushing.
2. Run from the repo root:

   ```bash
   bash .agents/skills/colab-e2e/run.sh               # scratch mode: checkpoints lost at stop
   bash .agents/skills/colab-e2e/run.sh --drive       # human mounts Drive; checkpoints persist
   ```

   Options: `-s NAME` (default `q2jepa-e2e`), `--gpu T4|L4|A100`, `--timeout SECONDS` (default 600).
   Agents use scratch mode: `--drive` waits for a human at the terminal.
3. The script stops the session on exit, including on failure. If it was killed, run
   `colab sessions` and `colab stop -s NAME` yourself.
4. Report: the script's final line, the benchmark tables and the metrics dict from
   `notebooks/colab_gpu_output.ipynb`, the GPU type, the commit SHA, and whether Drive or scratch mode was used.
   On failure, read the output notebook and `colab log -s NAME -n 50` before retrying.
