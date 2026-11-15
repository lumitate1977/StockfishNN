from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chess_ai import ROOT


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def blend_checkpoints(latest_path: Path, pretrained_path: Path, output_path: Path,
                      pretrained_weight: float, overwrite: bool = False) -> None:
    if not 0.0 <= pretrained_weight <= 1.0:
        raise ValueError("--pretrained-weight must be between 0 and 1")
    if not latest_path.is_file():
        raise FileNotFoundError(f"Latest checkpoint not found: {latest_path}")
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Human-pretrained checkpoint not found: {pretrained_path}")
    protected = {latest_path.resolve(), pretrained_path.resolve()}
    if output_path.resolve() in protected:
        raise ValueError("Output must be a new file; source checkpoints are protected.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Choose another path or pass --overwrite.")

    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if "model_state" not in latest or "model_state" not in pretrained:
        raise ValueError("Both inputs must be StockfishNN checkpoints containing model_state.")

    latest_state = latest["model_state"]
    pretrained_state = pretrained["model_state"]
    if latest_state.keys() != pretrained_state.keys():
        raise ValueError("Checkpoint model parameters do not match. Cannot safely blend different architectures.")

    blended = {}
    for key in latest_state:
        a, b = latest_state[key], pretrained_state[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f"Unexpected non-tensor model state entry: {key}")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Checkpoint tensor mismatch at {key}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        if a.is_floating_point() or a.is_complex():
            blended[key] = torch.lerp(a, b, pretrained_weight)
        else:
            # Integer BatchNorm counters and boolean buffers cannot be averaged.
            # Retain the value from latest.pt.
            blended[key] = a.clone()

    metadata = dict(latest.get("metadata", {}))
    metadata.update({
        "checkpoint_merge": "linear_weight_blend",
        "merge_latest_source": str(latest_path),
        "merge_human_pretrained_source": str(pretrained_path),
        "merge_latest_weight": 1.0 - pretrained_weight,
        "merge_human_pretrained_weight": pretrained_weight,
        "optimizer_state": "not_combined; a fresh optimizer will be used when training resumes",
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": blended, "metadata": metadata}, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blend latest.pt and human_pretrained.pt into a separate StockfishNN checkpoint."
    )
    parser.add_argument("--latest", default="models/latest.pt")
    parser.add_argument("--pretrained", default="models/human_pretrained.pt")
    parser.add_argument("--output", default="models/combined.pt")
    parser.add_argument("--pretrained-weight", type=float, default=0.5,
                        help="Contribution from human_pretrained.pt (0 to 1; default 0.5)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    latest_path = resolve_path(args.latest)
    pretrained_path = resolve_path(args.pretrained)
    output_path = resolve_path(args.output)
    blend_checkpoints(latest_path, pretrained_path, output_path,
                      args.pretrained_weight, args.overwrite)
    print("Checkpoint blend complete.")
    print(f"  latest.pt contribution: {1.0 - args.pretrained_weight:.0%}")
    print(f"  human_pretrained.pt contribution: {args.pretrained_weight:.0%}")
    print(f"  Saved combined checkpoint: {output_path}")
    print("Original checkpoints and replay data were not changed.")
    print("Note: this is parameter interpolation, not guaranteed to improve playing strength.")
    print("The combined checkpoint has no optimizer state; resumed training starts with a fresh optimizer.")


if __name__ == "__main__":
    main()
