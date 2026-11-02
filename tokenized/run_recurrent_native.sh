#!/usr/bin/env bash
# =============================================================================
# Recurrent-Native HGDM Training Launcher (8-Band HBS, TBPTT, MTP)
# =============================================================================
set -e

REPO_DIR="$HOME/htspc"
VENV_DIR="$HOME/htspc/.venv"
SESSION_NAME="rec_native"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing previous session '$SESSION_NAME'..."
    tmux kill-session -t "$SESSION_NAME"
fi

echo "================================================================="
echo "  HGDM RECURRENT-NATIVE TRAINING LAUNCHER"
echo "  Architecture: 8 Dyadic Bands, TBPTT, MTP, Lateral Inhibition"
echo "  Budget: 100M Tokens (~6,100 steps)"
echo "================================================================="

tmux new-session -d -s "$SESSION_NAME" \
    "source $VENV_DIR/bin/activate && cd $REPO_DIR && python -u tokenized/train_recurrent_native.py --tier pilot --batch_size 4 --accum_steps 4 --max_tokens 100000000 --save_dir checkpoints_tokenized/recurrent_native --save_every 500 --eval_every 250 2>&1 | tee train_rec_native.log"

echo "Recurrent-Native training successfully launched in tmux session '$SESSION_NAME'!"
echo "Log file: $REPO_DIR/train_rec_native.log"
echo "To monitor live: tmux attach -t $SESSION_NAME"
