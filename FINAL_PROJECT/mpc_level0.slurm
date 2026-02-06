#!/bin/bash
#SBATCH -t 03:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=flexvit_mpc_lvl01
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=genoa
#SBATCH --output=./logs/mpc_lvl0_%A.out
#SBATCH --error=./logs/mpc_lvl0_%A.err

mkdir -p logs

echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

# ---- (A) Load modules (Snellius) ----
module purge
module load 2023
module load Anaconda3/2023.07-2

# ---- (B) Activate conda env ----
source activate flexvit

# ---- (C) Paths ----
PROJECT_DIR="/gpfs/home2/nalmawaldi/FINAL_PROJECT/flexvit"
cd "$PROJECT_DIR" || exit 1

export CUDA_VISIBLE_DEVICES=""

# (optional) keep caches off GPFS
export TORCH_HOME="$TMPDIR/torch"
export XDG_CACHE_HOME="$TMPDIR/.cache"
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"

# ---- (D) Run MPC (level 0 only) ----
srun python -u run_mpc_cpu.py \
  --bw-mbps  \
  --levels 0 1 2 3 4 \
  --out-dir outputs/bw200_layers \
  --profile-layers \
  --profile-out-dir outputs/profiles/bw200_layers \
  --plain-cpu-profile-iters 5


