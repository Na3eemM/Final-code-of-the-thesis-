import os
import argparse
import secureD3_infer_crypten as exp  # your main module

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imagenet-root",
        default="/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    # Where to save MPC results CSV
    parser.add_argument("--out-dir", default="outputs")

    # Bandwidth cap (0 = no throttle)
    parser.add_argument(
        "--bw-mbps",
        type=float,
        default=0.0,
        help="Simulated bandwidth cap in Mbps for comm (0 = no throttling)",
    )

    # Enable per-layer profiling inside MPC + plain CPU profiling
    parser.add_argument(
        "--profile-layers",
        action="store_true",
        help="Enable per-layer timing (plain CPU + CrypTen module timing).",
    )
    parser.add_argument(
        "--profile-out-dir",
        default="outputs/profiles",
        help="Directory to write per-layer profiling CSVs.",
    )

    # ✅ NEW: name filters for CrypTen per-layer profiling
    parser.add_argument(
        "--profile-name-filters",
        nargs="+",
        default=["conv", "proj", "patch", "encoder", "layer", "head", "mlp", "attn", "attention"],
        help="Substrings to select CrypTen modules for layer profiling",
    )

    # Optional: how many iters for plain CPU profiling
    parser.add_argument(
        "--plain-cpu-profile-iters",
        type=int,
        default=5,
        help="Iterations for plain CPU per-layer profiling.",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # secureD3_infer_crypten.py expects this
    args.out_csv_mpc = os.path.join(args.out_dir, f"results_mpc_idx{args.index}.csv")

    # pass args into the global container used by run_party()
    exp.ARGS = args

    exp.run_party()

