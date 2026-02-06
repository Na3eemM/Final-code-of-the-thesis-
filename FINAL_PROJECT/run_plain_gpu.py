from secureD3_infer_crypten import run_plain_baseline  
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-root", default="/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder")
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--levels", type=int, nargs="+", default=[0,1,2,3,4])
    parser.add_argument("--plain-iters", type=int, default=10)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"results_plain_idx{args.index}.csv")

    run_plain_baseline(
        levels=args.levels,
        imagenet_root=args.imagenet_root,
        split=args.split,
        index=args.index,
        iters=args.plain_iters,
        out_csv=out_csv,
    )
