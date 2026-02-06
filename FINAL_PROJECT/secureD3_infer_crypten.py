"""
Author: <Naeem Almawaldi>
Project: <Your Thesis Title>
Institution: <Amsterdam University>
Year: <2026>

Description:
This file implements experimental code for evaluating multi-level FlexViT
inference under secure multi party computation MPC using the CrypTen framework.
The code includes:
- Plain (non-MPC) baseline inference and profiling
- MPC inference using CrypTen
- Communication cost measurement and bandwidth simulation
- Optional per layer and stage wise profiling

Inspiration and Related Work:
This implementation is inspired by and builds upon the following works and tools:
- CrypTen: Privacy preserving machine learning framework
- FlexiViT / Vision Transformer architectures
- PyTorch and torchvision utilities
- fvcore FLOP analysis tools
"""

import os
import math
import time
import argparse
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import crypten
import crypten.nn as cnn

from networks import flexvit
from networks.vit import ViTPrebuilt
from torchvision.datasets import ImageFolder


# FlexViT config for ImageNet
FLEXVIT_CONFIG = flexvit.ViTConfig(
    prebuilt=ViTPrebuilt.noprebuild,
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48),
)

CKPT_PATH = "../FlexViT_5Levels_cosine.pt"


def now() -> float:
    """High-resolution timer."""
    return time.perf_counter()


# CrypTen / ONNX friendly attention
class CrypTenFriendlyMHA(nn.Module):
    """
    Replacement for nn.MultiheadAttention (batch_first=True).
    Uses only matmul, softmax, reshape/transpose with explicit constants.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout

    def forward(self, query, key=None, value=None, **kwargs):
        x = query
        B, S, E = x.size(0), x.size(1), x.size(2)

        qkv = self.qkv(x)
        q = qkv[:, :, 0:E]
        k = qkv[:, :, E:2 * E]
        v = qkv[:, :, 2 * E:3 * E]

        H = self.num_heads
        D = self.head_dim

        q = q.reshape(B, S, H, D).transpose(1, 2)
        k = k.reshape(B, S, H, D).transpose(1, 2)
        v = v.reshape(B, S, H, D).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, E)
        out = self.proj(out)

        return out, None


def replace_mha_modules(model: nn.Module):
    """Replace nn.MultiheadAttention with CrypTenFriendlyMHA (for CrypTen conversion)."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            new_mha = CrypTenFriendlyMHA(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                dropout=0.0,
            )
            with torch.no_grad():
                if hasattr(child, "in_proj_weight") and child.in_proj_weight is not None:
                    new_mha.qkv.weight.copy_(child.in_proj_weight)
                else:
                    raise RuntimeError(
                        "Unsupported MultiheadAttention weight format.")
                new_mha.proj.weight.copy_(child.out_proj.weight)

            setattr(model, name, new_mha)
        else:
            replace_mha_modules(child)


# Communication logger
def install_comm_logger(bw_mbps: float = 0.0):
    """
    Measures:
      - bytes transferred
      - number of comm calls
      - time spent INSIDE comm primitives (lower bound)
      bw_mbps = 0 means no throttle
    """
    comm = crypten.communicator.get()
    counters = {"bytes": 0, "calls": 0, "comm_time_sec": 0.0}

    bw_bytes_per_sec = 0.0
    if bw_mbps and bw_mbps > 0:
        bw_bytes_per_sec = (bw_mbps * 1e6) / 8.0

    def _sizeof(obj):
        if torch.is_tensor(obj):
            return obj.nelement() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(_sizeof(x) for x in obj)
        return 0

    orig_send = comm.send
    orig_recv = comm.recv
    orig_all_reduce = comm.all_reduce

    def _maybe_throttle(nbytes: int):
        if bw_bytes_per_sec > 0 and nbytes > 0:
            time.sleep(nbytes / bw_bytes_per_sec)

    def send_wrapped(tensor, dst, *args, **kwargs):
        nbytes = _sizeof(tensor)
        counters["bytes"] += nbytes
        counters["calls"] += 1
        t0 = time.time()
        out = orig_send(tensor, dst, *args, **kwargs)
        counters["comm_time_sec"] += (time.time() - t0)
        _maybe_throttle(nbytes)
        return out

    def recv_wrapped(tensor, src=None, *args, **kwargs):
        nbytes = _sizeof(tensor)
        counters["bytes"] += nbytes
        counters["calls"] += 1
        t0 = time.time()
        out = orig_recv(tensor, src=src, *args, **kwargs)
        counters["comm_time_sec"] += (time.time() - t0)
        _maybe_throttle(nbytes)
        return out

    def all_reduce_wrapped(tensor, *args, **kwargs):
        nbytes = _sizeof(tensor)
        counters["bytes"] += 2 * nbytes
        counters["calls"] += 1
        t0 = time.time()
        out = orig_all_reduce(tensor, *args, **kwargs)
        counters["comm_time_sec"] += (time.time() - t0)
        _maybe_throttle(2 * nbytes)
        return out

    comm.send = send_wrapped
    comm.recv = recv_wrapped
    comm.all_reduce = all_reduce_wrapped
    return counters


# CSV writer pandas if available, otherwise just csv module
def _safe_write_csv(rows: List[Dict[str, Any]], out_csv: str, keys: Optional[List[str]] = None, tag: str = ""):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not rows:
        print(f"{tag} WARNING: no rows to write for {out_csv}", flush=True)
        return
    if keys is None:
        keys = list(rows[0].keys())

    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"{tag} wrote {out_csv}", flush=True)
        return
    except Exception:
        pass

    import csv
    try:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"{tag} wrote {out_csv} (csv module)", flush=True)
    except Exception as e:
        print(f"{tag} WARNING: could not write csv {out_csv}: {repr(e)}", flush=True)


# Model build / load
def build_flexvit(device: str, level: Optional[int] = None):
    model = FLEXVIT_CONFIG.make_model().to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)

    if level is not None:
        model.set_level_use(level)
        model = model.make_base_copy().to(device)

    model.eval()
    return model


# Dataset sample
def get_one_imagenet_sample(device: str, imagenet_root: str, split: str = "val", index: int = 0):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    dataset = ImageFolder(root=f"{imagenet_root}/{split}", transform=transform)
    x, y = dataset[index]
    return x.unsqueeze(0).to(device), int(y)


# FLOPs using fvcore
def try_compute_flops(model: nn.Module) -> Optional[float]:
    try:
        from fvcore.nn import FlopCountAnalysis
        import copy
        x = torch.randn(1, 3, 224, 224)
        model_cpu = copy.deepcopy(model).to("cpu")
        model_cpu.eval()
        flops = float(FlopCountAnalysis(model_cpu, x).total())
        return flops
    except Exception as e:
        print("[FLOPs] failed:", repr(e))
        return None


# Plain baseline GPU if we want to run it on GPU otherwise CPU wil also work
def run_plain_baseline(levels: List[int], imagenet_root: str, split: str, index: int,
                       iters: int, out_csv: str) -> List[Dict[str, Any]]:
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[plain] device={device}", flush=True)

    x, y_true = get_one_imagenet_sample(
        device=device, imagenet_root=imagenet_root, split=split, index=index)

    for lvl in levels:
        m = build_flexvit(device=device, level=lvl)
        flops = try_compute_flops(m)

        with torch.no_grad():
            for _ in range(3):
                _ = m(x)
            if device == "cuda":
                torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters):
                y = m(x)
            if device == "cuda":
                torch.cuda.synchronize()
        t1 = time.time()

        sec_per_iter = (t1 - t0) / float(iters)
        pred = int(y.argmax(dim=1).item())
        acc1 = 1.0 if pred == y_true else 0.0

        results.append({
            "mode": "plain",
            "device": device,
            "level": lvl,
            "index": index,
            "split": split,
            "sec_per_iter": sec_per_iter,
            "acc1": acc1,
            "pred": pred,
            "y_true": y_true,
            "comm_gb": 0.0,
            "calls": 0,
            "flops": flops if flops is not None else "",
        })

        print(
            f"[plain] level={lvl} sec/iter={sec_per_iter:.4f} acc1={acc1}", flush=True)

    _safe_write_csv(results, out_csv, keys=list(
        results[0].keys()) if results else None, tag="[plain]")
    return results


# Plain CPU per-layer profiling (PyTorch hooks)
def _find_modules_for_layer_profile(torch_model: nn.Module) -> Dict[str, nn.Module]:
    modules: Dict[str, nn.Module] = {}

    if hasattr(torch_model, "conv_proj"):
        modules["patch_embed_conv_proj"] = torch_model.conv_proj

    layers = None
    if hasattr(torch_model, "encoder") and hasattr(torch_model.encoder, "layers"):
        layers = torch_model.encoder.layers

    if layers is not None:
        for i, layer in enumerate(layers):
            modules[f"encoder_layer_{i}"] = layer

    if hasattr(torch_model, "heads"):
        modules["head"] = torch_model.heads
    elif hasattr(torch_model, "head"):
        modules["head"] = torch_model.head
    elif hasattr(torch_model, "fc"):
        modules["head"] = torch_model.fc

    return modules


def run_plain_cpu_layer_profile(
    levels: List[int],
    imagenet_root: str,
    split: str,
    index: int,
    iters: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cpu"
    print(f"[exp1_plain_cpu_layers] device={device}", flush=True)

    x, y_true = get_one_imagenet_sample(
        device=device, imagenet_root=imagenet_root, split=split, index=index)

    for lvl in levels:
        m = build_flexvit(device=device, level=lvl)
        m.eval()

        modules = _find_modules_for_layer_profile(m)
        if not modules:
            print(
                "[exp1_plain_cpu_layers] WARNING: no modules found for profiling", flush=True)
            continue

        times = {name: 0.0 for name in modules.keys()}
        calls = {name: 0 for name in modules.keys()}
        starts = {}

        def pre_hook(name):
            def _pre(_mod, _inp):
                starts[name] = time.perf_counter()
            return _pre

        def post_hook(name):
            def _post(_mod, _inp, _out):
                dt = time.perf_counter() - starts.get(name, time.perf_counter())
                times[name] += dt
                calls[name] += 1
            return _post

        hooks = []
        for name, mod in modules.items():
            hooks.append(mod.register_forward_pre_hook(pre_hook(name)))
            hooks.append(mod.register_forward_hook(post_hook(name)))

        with torch.no_grad():
            for _ in range(2):
                _ = m(x)

        with torch.no_grad():
            for _ in range(iters):
                y = m(x)

        pred = int(y.argmax(dim=1).item())
        acc1 = 1.0 if pred == y_true else 0.0

        for h in hooks:
            h.remove()

        rows = []
        for name in sorted(times.keys()):
            rows.append({
                "mode": "plain_cpu_layer_profile",
                "level": lvl,
                "index": index,
                "split": split,
                "module": name,
                "total_time_sec": times[name],
                "avg_time_per_infer_sec": times[name] / max(1, iters),
                "calls": calls[name],
                "acc1": acc1,
                "pred": pred,
                "y_true": y_true,
            })

        out_csv = os.path.join(
            out_dir, f"plain_cpu_layers_level{lvl}_idx{index}.csv")
        _safe_write_csv(rows, out_csv, keys=list(
            rows[0].keys()) if rows else None, tag="[exp1_plain_cpu_layers]")


# CrypTen model build
def build_crypten_model_for_level(level: int):
    device = "cpu"
    torch_model = build_flexvit(device=device, level=level)
    replace_mha_modules(torch_model)

    dummy = torch.randn(1, 3, 224, 224)
    crypten_model = cnn.from_pytorch(torch_model, dummy_input=dummy)
    crypten_model.encrypt()
    crypten_model.eval()
    return crypten_model


# Global args container (used inside multiprocess)
ARGS = None


@crypten.mpc.run_multiprocess(world_size=2)
def run_party():
    global ARGS
    args = ARGS
    print("[mpc] entered run_party()", flush=True)

    if args is None:
        raise RuntimeError(
            "ARGS is None. You must set exp.ARGS before calling run_party().")

    comm = crypten.communicator.get()
    rank = comm.get_rank()

    def r0print(*a, **k):
        if rank == 0:
            print(*a, **k, flush=True)

    comm_counters = install_comm_logger(bw_mbps=args.bw_mbps)

    def barrier():
        if hasattr(comm, "barrier"):
            comm.barrier()
        else:
            t = torch.tensor([1], dtype=torch.long)
            comm.all_reduce(t)

    # Input
    if rank == 0:
        x_plain, y_true = get_one_imagenet_sample(
            device="cpu",
            imagenet_root=args.imagenet_root,
            split=args.split,
            index=args.index,
        )
    else:
        x_plain = torch.empty(1, 3, 224, 224)
        y_true = -1

    x_enc = crypten.cryptensor(x_plain, src=0)
    if rank == 0:
        r0print(
            f"[mpc] MPC ok: world_size=2, x_enc_type={type(x_enc).__name__}, src=0")

    # Build CrypTen models (one per level) + measure build time per level
    crypten_models = {}
    build_times_sec = {}

    for lvl in args.levels:
        barrier()
        t_build0 = now()
        crypten_models[lvl] = build_crypten_model_for_level(lvl)
        t_build1 = now()
        barrier()

        if rank == 0:
            build_times_sec[lvl] = float(t_build1 - t_build0)

    # Loop over levels: timed forward + comm metrics
    # Also write stage CSV per level if --profile-layers
    results = []
    output_enc = None

    do_profile = bool(getattr(args, "profile_layers", False))

    for lvl in args.levels:
        comm_counters["bytes"] = 0
        comm_counters["calls"] = 0
        comm_counters["comm_time_sec"] = 0.0

        barrier()
        t0 = time.time()
        output_enc = crypten_models[lvl](x_enc)
        barrier()
        elapsed = time.time() - t0

        if rank == 0:
            gb = comm_counters["bytes"] / (1024 ** 3)
            r0print(
                f"[mpc] level={lvl} time={elapsed:.2f}s comm≈{gb:.2f}GB calls={comm_counters['calls']}")

            # Main results row
            results.append({
                "mode": "mpc",
                "device": "cpu",
                "level": lvl,
                "index": args.index,
                "split": args.split,
                "sec_per_iter": float(elapsed),
                "comm_gb": float(gb),
                "calls": int(comm_counters["calls"]),
                "comm_time_sec": float(comm_counters["comm_time_sec"]),
                "comm_fraction": (float(comm_counters["comm_time_sec"]) / float(elapsed) if elapsed > 0 else ""),
                "bw_mbps": float(args.bw_mbps),
                "flops": "",
                "acc1": "",
                "pred": "",
                "y_true": int(y_true),
            })

            # Stage breakdown per level
            if do_profile:
                os.makedirs(args.profile_out_dir, exist_ok=True)
                stage_rows = [{
                    "mode": "mpc_stage_profile",
                    "bw_mbps": float(args.bw_mbps),
                    "level": lvl,
                    "index": args.index,
                    "split": args.split,
                    "build_time_sec": float(build_times_sec.get(lvl, 0.0)),
                    "forward_time_sec": float(elapsed),
                    "comm_time_sec": float(comm_counters["comm_time_sec"]),
                    "non_comm_time_sec": float(elapsed - comm_counters["comm_time_sec"]),
                    "comm_gb": float(gb),
                    "calls": int(comm_counters["calls"]),
                }]
                out_csv_stage = os.path.join(
                    args.profile_out_dir,
                    f"mpc_stage_bw{int(args.bw_mbps)}_level{lvl}_idx{args.index}.csv"
                )
                _safe_write_csv(stage_rows, out_csv_stage, keys=list(
                    stage_rows[0].keys()), tag="[mpc_stage]")

    # Save the main MPC CSV (rank0)
    if rank == 0:
        if len(results) == 0:
            r0print("[mpc] WARNING: no results rows to write.")
        else:
            _safe_write_csv(results, args.out_csv_mpc,
                            keys=list(results[0].keys()), tag="[mpc]")

    # decrypt (avoid crashes)
    barrier()

    try_decrypt = bool(getattr(args, "try_decrypt", False))
    if rank == 0 and output_enc is not None and try_decrypt:
        try:
            out = output_enc.get_plain_text()
            pred = int(out.argmax(dim=1).item())
            acc1 = 1.0 if pred == int(y_true) else 0.0
            r0print(
                f"[mpc] final_pred={pred} y_true={int(y_true)} acc1={acc1}")

            if len(results) > 0:
                results[-1]["pred"] = pred
                results[-1]["acc1"] = acc1
                _safe_write_csv(results, args.out_csv_mpc,
                                keys=list(results[0].keys()), tag="[mpc]")

        except Exception as e:
            r0print(f"[mpc] WARNING: decrypt failed: {repr(e)}")

    barrier()


def main():
    global ARGS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet-root", default="/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder")
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--levels", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--out-dir", default="outputs")

    parser.add_argument("--bw-mbps", type=float, default=0.0,
                        help="Simulated bandwidth cap in Mbps for comm (0 = no throttle).")

    parser.add_argument("--run-plain-gpu", action="store_true",
                        help="Also run plain baseline (GPU if available) and write results_plain_idx*.csv")
    parser.add_argument("--plain-iters", type=int, default=10)

    parser.add_argument("--profile-layers", action="store_true",
                        help="Enable EXP1 (plain CPU per-layer) + EXP2 (MPC stage breakdown CSVs).")
    parser.add_argument("--plain-cpu-profile-iters", type=int, default=5,
                        help="Iterations for EXP1 (plain CPU per-layer).")
    parser.add_argument("--profile-out-dir", default="outputs/profiles",
                        help="Directory for profiling CSVs (EXP1 + EXP2-stage).")

    parser.add_argument("--try-decrypt", action="store_true",
                        help="Attempt decrypt at end (optional; can timeout on cluster).")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.out_csv_mpc = os.path.join(
        args.out_dir, f"results_mpc_idx{args.index}.csv")
    args.out_csv_plain = os.path.join(
        args.out_dir, f"results_plain_idx{args.index}.csv")

    ARGS = args

    if args.run_plain_gpu:
        run_plain_baseline(
            levels=args.levels,
            imagenet_root=args.imagenet_root,
            split=args.split,
            index=args.index,
            iters=args.plain_iters,
            out_csv=args.out_csv_plain,
        )

    if args.profile_layers:
        run_plain_cpu_layer_profile(
            levels=args.levels,
            imagenet_root=args.imagenet_root,
            split=args.split,
            index=args.index,
            iters=args.plain_cpu_profile_iters,
            out_dir=args.profile_out_dir,
        )

    run_party()


if __name__ == "__main__":
    main()
