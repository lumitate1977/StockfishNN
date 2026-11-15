from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chess_ai import ROOT


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_model_spec(value: str) -> tuple[Path, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Each --model must be PATH=WEIGHT (example: models/latest.pt=50)")
    path_text, weight_text = value.rsplit("=", 1)
    try:
        weight = float(weight_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid weight in {value!r}") from exc
    if not path_text.strip() or weight < 0:
        raise argparse.ArgumentTypeError("Model path must be non-empty and weight cannot be negative")
    return resolve_path(path_text.strip()), weight


def blend_checkpoints(models: list[tuple[Path, float]], output_path: Path,
                      overwrite: bool = False) -> None:
    if len(models) < 2:
        raise ValueError("Select at least two checkpoints to combine.")
    total = sum(weight for _, weight in models)
    if total <= 0 or abs(total - 100.0) > 0.01:
        raise ValueError(f"Model percentages must add up to 100 (currently {total:.2f}).")
    normalized = [(path, weight / 100.0) for path, weight in models if weight > 0]
    if len(normalized) < 2:
        raise ValueError("At least two selected models must have a non-zero percentage.")
    paths = [path.resolve() for path, _ in normalized]
    if len(set(paths)) != len(paths):
        raise ValueError("The same checkpoint was selected more than once.")
    if output_path.resolve() in set(paths):
        raise ValueError("Output must be a new file; source checkpoints are protected.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Choose another path or pass --overwrite.")

    checkpoints = []
    for path, weight in normalized:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state" not in checkpoint:
            raise ValueError(f"Not a StockfishNN checkpoint (missing model_state): {path}")
        checkpoints.append((path, weight, checkpoint))

    reference = checkpoints[0][2]["model_state"]
    for path, _weight, checkpoint in checkpoints[1:]:
        state = checkpoint["model_state"]
        if state.keys() != reference.keys():
            raise ValueError(f"Model parameter keys differ: {path}")
        for key in reference:
            a, b = reference[key], state[key]
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError(f"Unexpected non-tensor model state entry: {key}")
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Checkpoint tensor mismatch at {key} in {path}")

    # For non-floating buffers (e.g. BatchNorm counters), copy from the
    # highest-weight source because integer/bool values cannot be averaged.
    anchor_state = max(checkpoints, key=lambda item: item[1])[2]["model_state"]
    blended = {}
    for key, reference_tensor in reference.items():
        if reference_tensor.is_floating_point() or reference_tensor.is_complex():
            result = torch.zeros_like(reference_tensor)
            for _path, weight, checkpoint in checkpoints:
                result.add_(checkpoint["model_state"][key], alpha=weight)
            blended[key] = result
        else:
            blended[key] = anchor_state[key].clone()

    metadata = dict(checkpoints[0][2].get("metadata", {}))
    metadata.update({
        "checkpoint_merge": "weighted_multi_model_parameter_blend",
        "merge_sources": [{"path": str(path), "weight_percent": weight * 100.0}
                          for path, weight, _checkpoint in checkpoints],
        "optimizer_state": "not_combined; a fresh optimizer will be used when training resumes",
        "merge_warning": "Parameter interpolation is not a knowledge-preserving merge and may reduce playing strength.",
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": blended, "metadata": metadata}, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a weighted blend of two or more compatible StockfishNN checkpoints."
    )
    parser.add_argument("--model", action="append", type=parse_model_spec,
                        help="Checkpoint and percentage, repeat for each model: --model models/latest.pt=50")
    parser.add_argument("--output", default="models/combined.pt")
    parser.add_argument("--overwrite", action="store_true")
    # Backward-compatible two-model arguments.
    parser.add_argument("--latest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pretrained", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pretrained-weight", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.model:
        if args.latest or args.pretrained or args.pretrained_weight is not None:
            parser.error("Use either repeated --model arguments or the legacy two-model arguments, not both.")
        models = args.model
    else:
        latest = resolve_path(args.latest or "models/latest.pt")
        pretrained = resolve_path(args.pretrained or "models/human_pretrained.pt")
        pretrained_weight = 50.0 if args.pretrained_weight is None else args.pretrained_weight * 100.0
        if not 0 <= pretrained_weight <= 100:
            parser.error("--pretrained-weight must be between 0 and 1")
        models = [(latest, 100.0 - pretrained_weight), (pretrained, pretrained_weight)]

    output = resolve_path(args.output)
    blend_checkpoints(models, output, args.overwrite)
    print("Checkpoint blend complete.")
    for path, weight in models:
        if weight:
            print(f"  {path}: {weight:.2f}%")
    print(f"Saved combined checkpoint: {output}")
    print("Original checkpoints and replay data were not changed.")
    print("Warning: parameter interpolation is not guaranteed to improve playing strength.")
    print("Optimizer state is not combined; resumed training creates a fresh optimizer.")


if __name__ == "__main__":
    main()
