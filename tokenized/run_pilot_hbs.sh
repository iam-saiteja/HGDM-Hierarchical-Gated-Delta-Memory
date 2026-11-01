#!/usr/bin/env bash
set -e

REPO_DIR="$HOME/htspc"
VENV_DIR="$HOME/htspc/.venv"
SESSION_NAME="hbs_pilot"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
fi

echo "Launching HBS Pilot (10M tokens) in tmux session '$SESSION_NAME'..."

tmux new-session -d -s "$SESSION_NAME" \
    "source $VENV_DIR/bin/activate && cd $REPO_DIR && python -u tokenized/train.py --model hgdm --use_hbs --tier pilot --max_tokens 10000000 --save_dir checkpoints_tokenized/pilot_hbs 2>&1 | tee hbs_pilot.log"

echo "HBS Pilot successfully launched!"
echo "Log file: $REPO_DIR/hbs_pilot.log"
