from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chess_ai import ACTION_SIZE, ROOT, ChessNet, encode_board, load_config, move_to_index


DEFAULT_THEMES = ("mateIn1", "mateIn2", "mateIn3")


def parse_puzzle(row: dict[str, str], skip_setup_move: bool, allowed_themes: set[str]):
    fen = (row.get("FEN") or row.get("fen") or "").strip()
    moves_text = (row.get("Moves") or row.get("moves") or "").strip()
    themes = set((row.get("Themes") or row.get("themes") or "").split())
    if not fen or not moves_text or not themes.intersection(allowed_themes):
        return None

    try:
        board = chess.Board(fen)
        moves = moves_text.split()
        parsed = [chess.Move.from_uci(token) for token in moves]
        if skip_setup_move:
            if not parsed:
                return None
            setup = parsed.pop(0)
            if setup not in board.legal_moves:
                return None
            board.push(setup)

        solver = board.turn
        samples = []
        for move in parsed:
            if move not in board.legal_moves:
                return None
            if board.turn == solver:
                legal = [move_to_index(candidate) for candidate in board.legal_moves]
                samples.append((encode_board(board), move_to_index(move), legal))
            board.push(move)

        # Only train on complete mating lines; don't teach a false mate target.
        if not board.is_checkmate() or not samples:
            return None
        return samples
    except (ValueError, AssertionError):
        return None


def load_dataset(path: Path, max_puzzles: int, skip_setup_move: bool, allowed_themes: set[str]):
    positions, actions, legal_lists = [], [], []
    puzzles_seen = puzzles_used = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not ({"FEN", "fen"} & set(reader.fieldnames)) or not ({"Moves", "moves"} & set(reader.fieldnames)):
            raise RuntimeError("Puzzle CSV needs FEN, Moves, and Themes columns (Lichess puzzle CSV format).")
        for row in reader:
            puzzles_seen += 1
            samples = parse_puzzle(row, skip_setup_move, allowed_themes)
            if samples:
                puzzles_used += 1
                for position, action, legal in samples:
                    positions.append(position)
                    actions.append(action)
                    legal_lists.append(legal)
            if max_puzzles > 0 and puzzles_seen >= max_puzzles:
                break

    if not positions:
        raise RuntimeError(
            "No complete mate puzzles were usable. Check the CSV, theme names, and --skip-setup-move setting."
        )
    offsets = [0]
    legal_flat = []
    for legal in legal_lists:
        legal_flat.extend(legal)
        offsets.append(len(legal_flat))
    return (
        np.stack(positions).astype(np.uint8, copy=False),
        np.asarray(actions, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(legal_flat, dtype=np.int32),
        puzzles_seen,
        puzzles_used,
    )


def evaluate_policy(model, positions, actions, offsets, legal_flat, device, batch_size):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            end = min(start + batch_size, len(positions))
            x_np = positions[start:end]
            rows = end - start
            counts = offsets[start + 1:end + 1] - offsets[start:end]
            row_ids = np.repeat(np.arange(rows, dtype=np.int64), counts)
            legal_ids = np.concatenate([
                legal_flat[offsets[i]:offsets[i + 1]] for i in range(start, end)
            ])
            mask = torch.zeros((rows, ACTION_SIZE), dtype=torch.bool)
            mask[row_ids, legal_ids] = True
            x = torch.from_numpy(x_np.astype(np.float32, copy=False)).to(device)
            logits, _ = model(x)
            predictions = logits.masked_fill(~mask.to(device), -1e9).argmax(dim=1).cpu().numpy()
            correct += int((predictions == actions[start:end]).sum())
            total += rows
    return correct / max(1, total)


def train_epoch(model, optimizer, positions, actions, offsets, legal_flat, device, batch_size, rng):
    order = rng.permutation(len(positions))
    model.train()
    loss_total = 0.0
    batches = 0
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        rows = len(ids)
        starts, ends = offsets[ids], offsets[ids + 1]
        counts = ends - starts
        row_ids = np.repeat(np.arange(rows, dtype=np.int64), counts)
        legal_ids = np.concatenate([legal_flat[a:b] for a, b in zip(starts, ends)])
        mask_np = np.zeros((rows, ACTION_SIZE), dtype=np.bool_)
        mask_np[row_ids, legal_ids] = True

        x = torch.from_numpy(positions[ids].astype(np.float32, copy=False)).to(device)
        targets = torch.from_numpy(actions[ids]).to(device=device, dtype=torch.long)
        mask = torch.from_numpy(mask_np).to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.masked_fill(~mask, -1e9), targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_total += float(loss.detach().item())
        batches += 1
    return loss_total / max(1, batches)


def main():
    parser = argparse.ArgumentParser(
        description="Supervised checkmate-puzzle training for StockfishNN (Lichess puzzle CSV)."
    )
    parser.add_argument("--puzzles", default="data/puzzles/lichess_db_puzzle.csv", help="Lichess puzzle CSV file")
    parser.add_argument("--base-checkpoint", default="models/human_pretrained.pt")
    parser.add_argument("--output", default="models/mate_pretrained.pt")
    parser.add_argument("--themes", default="mateIn1,mateIn2,mateIn3", help="Comma-separated Lichess themes")
    parser.add_argument("--max-puzzles", type=int, default=50000, help="0 means all matching puzzles")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.00002)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-skip-setup-move", action="store_true", help="Use if your CSV FEN is already after the opponent setup move")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_puzzles < 0:
        parser.error("epochs and batch-size must be positive; max-puzzles cannot be negative")
    if args.learning_rate <= 0 or not 0 <= args.validation_fraction < 0.5:
        parser.error("learning-rate must be positive and validation-fraction must be in [0, 0.5)")
    themes = {item.strip() for item in args.themes.split(",") if item.strip()}
    if not themes:
        parser.error("At least one theme is required")

    def resolve(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    data_path = resolve(args.puzzles)
    base_path = resolve(args.base_checkpoint)
    output_path = resolve(args.output)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Puzzle CSV not found: {data_path}\nDownload the Lichess puzzle database and place its CSV here."
        )
    if not base_path.is_file():
        config = load_config()
        fallback = resolve(config.get("model_path", "models/latest.pt"))
        if not fallback.is_file():
            raise FileNotFoundError(f"Base checkpoint not found: {base_path} (or fallback {fallback})")
        print(f"Requested base checkpoint missing; using configured checkpoint: {fallback}")
        base_path = fallback
    if output_path.resolve() in {base_path.resolve(), resolve("models/latest.pt").resolve(),
                                  resolve("models/human_pretrained.pt").resolve()}:
        parser.error("Output must be a new checkpoint; source checkpoints are protected.")
    if output_path.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output_path}. Choose another path or pass --overwrite.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Reading mate puzzles from: {data_path}")
    data = load_dataset(data_path, args.max_puzzles, not args.no_skip_setup_move, themes)
    positions, actions, offsets, legal_flat, puzzles_seen, puzzles_used = data
    print(f"Loaded {puzzles_used:,} complete mate puzzles from {puzzles_seen:,} CSV rows.")
    print(f"Training samples (solver-to-move positions): {len(positions):,}")
    print(f"Device: {device} | base checkpoint: {base_path}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(positions))
    val_count = int(len(order) * args.validation_fraction)
    if val_count:
        val_ids, train_ids = order[:val_count], order[val_count:]
    else:
        val_ids, train_ids = np.array([], dtype=np.int64), order
    train_data = (positions[train_ids], actions[train_ids])
    train_offsets_parts = [offsets[i:i + 2] for i in train_ids]
    train_legal_lists = [legal_flat[a:b] for a, b in
                         ((offsets[i], offsets[i + 1]) for i in train_ids)]
    train_offsets = np.zeros(len(train_ids) + 1, dtype=np.int64)
    train_offsets[1:] = np.cumsum([len(x) for x in train_legal_lists])
    train_legal = np.concatenate(train_legal_lists).astype(np.int32, copy=False)

    checkpoint = torch.load(base_path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        raise RuntimeError(f"Invalid StockfishNN checkpoint (missing model_state): {base_path}")
    model = ChessNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    print(f"Initial validation top-1 accuracy: " +
          (f"{evaluate_policy(model, positions[val_ids], actions[val_ids], offsets[val_ids.tolist() + [int(val_ids[-1]) + 1]] if False else np.array([0]), legal_flat, device, args.batch_size):.4f}" if False else "will be measured after training"))
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer, train_data[0], train_data[1],
                           train_offsets, train_legal, device, args.batch_size, rng)
        print(f"Epoch {epoch}/{args.epochs} | policy loss {loss:.5f}")

    val_accuracy = None
    if val_count:
        val_legal_lists = [legal_flat[offsets[i]:offsets[i + 1]] for i in val_ids]
        val_offsets = np.zeros(len(val_ids) + 1, dtype=np.int64)
        val_offsets[1:] = np.cumsum([len(x) for x in val_legal_lists])
        val_legal = np.concatenate(val_legal_lists).astype(np.int32, copy=False)
        val_accuracy = evaluate_policy(model, positions[val_ids], actions[val_ids],
                                       val_offsets, val_legal, device, args.batch_size)
        print(f"Validation top-1 puzzle-move accuracy: {val_accuracy:.2%}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.update({
        "pretraining": "checkmate_puzzles",
        "puzzle_source": str(data_path),
        "puzzle_themes": sorted(themes),
        "puzzles_seen": puzzles_seen,
        "puzzles_used": puzzles_used,
        "training_positions": int(len(train_ids)),
        "validation_positions": int(val_count),
        "validation_top1_accuracy": val_accuracy,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "base_checkpoint": str(base_path),
        "architecture": "ChessNet-18plane-policy-value-v1",
    })
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metadata": metadata}, output_path)
    print(f"Checkmate-trained checkpoint saved: {output_path}")
    print(f"Elapsed: {time.time() - started:.1f}s")
    print("Original checkpoints and replay data were not modified.")


if __name__ == "__main__":
    main()
