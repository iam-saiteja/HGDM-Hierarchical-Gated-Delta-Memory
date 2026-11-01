#!/usr/bin/env bash
# =============================================================================
# Full Pipeline: Chinchilla Pretraining (619M Tokens) -> SFT (500 Steps) -> Head-to-Head Arena
# Architecture: Pure Recurrent HGDM with Hierarchical Band Search (HBS)
# Zero Attention Matrices, Zero KV-Cache, O(1) Memory
# =============================================================================
set -e

REPO_DIR="$HOME/htspc"
VENV_DIR="$HOME/htspc/.venv"
SESSION_NAME="hbs_full_run"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists! Killing old session..."
    tmux kill-session -t "$SESSION_NAME"
fi

mkdir -p checkpoints_tokenized/full_hbs

echo "================================================================="
echo "  STARTING FULL HBS CHINCHILLA PRETRAINING PIPELINE"
echo "  Tmux Session: $SESSION_NAME"
echo "  Log File:     $REPO_DIR/hbs_full_train.log"
echo "================================================================="

tmux new-session -d -s "$SESSION_NAME" \
    "source $VENV_DIR/bin/activate && cd $REPO_DIR && bash -c '
        echo \"[Stage 1/3] Full Chinchilla Pretraining Starting at \$(date)\" && \
        python -u tokenized/train.py --model hgdm --use_hbs --tier pilot --save_dir checkpoints_tokenized/full_hbs 2>&1 | tee hbs_full_train.log && \
        echo \"[Stage 2/3] Conversational SFT Starting at \$(date)\" && \
        python -u tokenized/train_sft.py --model_type hgdm --base_ckpt checkpoints_tokenized/full_hbs/ckpt_hgdm.pt --output_ckpt checkpoints_tokenized/full_hbs/ckpt_hgdm_sft.pt --steps 500 2>&1 | tee -a hbs_full_train.log && \
        echo \"[Stage 3/3] Comprehensive Head-to-Head Arena Starting at \$(date)\" && \
        python -u tokenized/run_comprehensive_arena.py --hgdm_ckpt checkpoints_tokenized/full_hbs/ckpt_hgdm_sft.pt --transformer_ckpt checkpoints_tokenized/ckpt_transformer_sft.pt 2>&1 | tee hbs_arena_results.log && \
        echo \"[Pipeline Finished at \$(date)] ALL STAGES COMPLETE!\" | tee -a hbs_full_train.log
    '"

echo "Pipeline launched successfully in background tmux session '$SESSION_NAME'!"
echo "To monitor live logs in real time: tail -f $REPO_DIR/hbs_full_train.log"
