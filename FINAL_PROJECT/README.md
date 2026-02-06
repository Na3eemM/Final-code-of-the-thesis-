# Dynamic FlexViT Secure Inference with CrypTen (2-Party MPC)

This repository provides an experimental framework for **secure and inference**
**FlexViT** models.

First of all you need to clone Flexvit from
https://gitlab.com/R0b4/flexvit

Then, download the ImageNet pretrained checkpoint (FlexViT_5Levels_cosine.pt) 
from the link I sent you via Gmail with the final version of the thesis, 
and place it in the same directory as SecureD3_infer_crypten.py.


**`secureD3_infer_crypten.py`**

users are expected to run experiments via the provided runner scripts not by
calling the main file directly.


## How to run the code

- `secureD3_infer_crypten.py`
  - Contains all experiment logic
  - Model construction, profiling, MPC, logging
  - Not intended to be modified or partially imported

- **Runner scripts (REQUIRED for normal use):**
  - `run_plain_gpu` → plain PyTorch baseline (GPU / CPU)
  - and you need to run this on snelluis to get GPU works
   " srun --partition=gpu_a100 --gres=gpu:1 --time=01:00:00 --pty bash "

  - `run_mpc_cpu.py` → secure inference (CrypTen MPC + CPU profiling)
    " on sneluis you can use the script mpc_level0.slurm to submit a job

you can look at the results of my experments in the output folder

Alos you need to take a look at the requirments.txt file
