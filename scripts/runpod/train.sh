#!/bin/bash
# Start Hemmingway-1 training in a detached tmux session (through lf_train.py,
# which lets LlamaFactory's qwen3_8 template run without an image processor), or resume it: the
# yaml keeps overwrite_output_dir false, so relaunching picks up the last
# checkpoint. Window "train" runs LlamaFactory (log in train.log), window
# "archive" copies each checkpoint's adapter to /workspace/adapters.
#
#   train.sh           full run
#   train.sh --smoke   5 steps with a save and an eval, into saves/smoke
#
# tmux attach -t train (detach: Ctrl-b d). Scroll with the mouse wheel, or
# Ctrl-b [ then PgUp/PgDn (q to leave).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN_DIR=/workspace/russell_training
YAML=hemmingway1_27b_lora.yaml
SAVES=$RUN_DIR/saves/Hemmingway-1/lora/russell
ARCHIVE=/workspace/adapters
# The datasets cache chmods its files, which the network volume refuses, so it
# stays on the container disk (it's small and rebuilt in a minute).
ENV="source /root/venv/bin/activate && export HF_HOME=/workspace/huggingface_cache \
HF_DATASETS_CACHE=/root/.cache/huggingface/datasets"

# Mouse-wheel scrolling and a long scrollback. history-limit only applies to
# windows created after it's set, so load it before the session starts.
grep -q "history-limit" ~/.tmux.conf 2>/dev/null || cat >> ~/.tmux.conf <<'CONF'
set -g mouse on
set -g history-limit 100000
CONF
tmux source-file ~/.tmux.conf 2>/dev/null || true

if tmux has-session -t train 2>/dev/null; then
    echo "tmux session 'train' already exists: tmux attach -t train"
    exit 1
fi

if [ "${1:-}" = "--smoke" ]; then
    # Checks memory, a save and an eval before committing to the long run.
    tmux new-session -d -s train -n train -c "$RUN_DIR" \
        "bash -c '$ENV && python $REPO/scripts/runpod/lf_train.py $YAML max_steps=5 save_steps=5 eval_steps=5 \
         logging_steps=1 output_dir=saves/smoke overwrite_output_dir=true 2>&1 | tee smoke.log; exec bash'"
    echo "Smoke test running: tmux attach -t train"
    exit 0
fi

tmux new-session -d -s train -n train -c "$RUN_DIR" \
    "bash -c '$ENV && python $REPO/scripts/runpod/lf_train.py $YAML 2>&1 | tee -a train.log; exec bash'"
tmux new-window -t train -n archive \
    "bash $REPO/scripts/runpod/archive_adapters.sh $SAVES $ARCHIVE"
echo "Training in tmux session 'train': tmux attach -t train"
