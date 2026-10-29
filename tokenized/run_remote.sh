#!/usr/bin/env bash
# =============================================================================
# Run Chinchilla Tokenized Arena Training in a Persistent TMUX Session on GPU
# =============================================================================
set -e

REPO_DIR="$HOME/htspc"
VENV_DIR="$HOME/htspc/.venv"
SESSION_NAME="token_arena"

echo "================================================================="
echo "  HGDM vs. TRANSFORMER: CHINCHILLA ARENA LAUNCHER"
echo "================================================================="

cd "$REPO_DIR"

# Verify virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment via uv..."
    ~/.local/bin/uv venv "$VENV_DIR" --python 3.12
    ~/.local/bin/uv pip install --python "$VENV_DIR" torch transformers datasets triton
fi

source "$VENV_DIR/bin/activate"

# Check if tmux session already running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists!"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

echo "Launching training in background tmux session: '$SESSION_NAME'..."

tmux new-session -d -s "$SESSION_NAME" \
    "source $VENV_DIR/bin/activate && cd $REPO_DIR && python tokenized/train.py --tier pilot --model both --dataset HuggingFaceTB/smoltalk 2>&1 | tee train_arena.log"

echo "================================================================="
echo "Training successfully launched in tmux!"
echo "To view live training output:"
echo "    tmux attach -t $SESSION_NAME"
echo "To detach from tmux without stopping training:"
echo "    Press Ctrl+b, then d"
echo "To view the log file:"
echo "    tail -f $REPO_DIR/train_arena.log"
echo "================================================================="
