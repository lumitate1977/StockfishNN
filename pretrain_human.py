from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

from chess_ai import ACTION_SIZE, ROOT, ChessNet, encode_board, load_config, move_to_index


def open_pgn(path: Path) -> TextIO:
    if path.name.lower().endswith(".pgn.gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open("r", encoding="utf-8-sig", errors="replace")


def find_pgn_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.name.lower().endswith((".pgn", ".pgn.gz")) else []
    if not source.is_dir():
        return []
    return sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.name.lower().endswith((".pgn", ".pgn.gz"))
    )


def game_value(result: str, turn: chess.Color) -> float | None:
    if result == "1-0":
        white_value = 1.0
    elif result == "0-1":
        white_value = -1.0
    elif result == "1/2-1/2":
        white_value = 0.0
    else:
        return None
    return white_value if turn == chess.WHITE else -white_value


def collect_positions(
    files: list[Path],
    max_games: int,
    max_positions: int,
    min_elo: int,
) -> tuple[np.ndarray, ...]:
    positions: list[np.ndarray] = []
    actions: list[int] = []
    values: list[float] = []
    offsets = [0]
    legal_flat: list[int] = []
    games_seen = 0
    games_used = 0

    for path in files:
        print(f"Reading: {path}")
        with open_pgn(path) as pgn_file:
            while len(positions) < max_positions:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games_seen += 1
                if max_games > 0 and games_seen > max_games:
                    break

                result = game.headers.get("Result", "*")
                if result not in ("1-0", "0-1", "1/2-1/2"):
                    continue

                if min_elo > 0:
                    try:
                        white_elo = int(game.headers.get("WhiteElo", "0"))
                        black_elo = int(game.headers.get("BlackElo", "0"))
                    except ValueError:
                        continue
                    if min(white_elo, black_elo) < min_elo:
                        continue

                board = game.board()
                game_added = 0
                for move in game.mainline_moves():
                    legal_moves = list(board.legal_moves)
                    if move not in legal_moves:
                        break
                    value = game_value(result, board.turn)
                    if value is None:
                        break
                    legal_indices = [move_to_index(candidate) for candidate in legal_moves]
                    target_action = move_to_index(move)
                    positions.append(encode_board(board))
                    actions.append(target_action)
                    values.append(value)
                    legal_flat.extend(legal_indices)
                    offsets.append(len(legal_flat))
                    game_added += 1
                    board.push(move)
                    if len(positions) >= max_positions:
                        break
                if game_added:
                    games_used += 1

            if max_games > 0 and games_seen >= max_games:
                break
            if len(positions) >= max_positions:
                break

    if not positions:
        raise RuntimeError(
            "No usable positions were found. Check that the source contains completed PGN games."
        )

    return (
        np.stack(positions).astype(np.uint8, copy=False),
        np.asarray(actions, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(legal_flat, dtype=np.int32),
        np.asarray([games_seen, games_used], dtype=np.int64),
    )


def train_epoch(
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    positions: np.ndarray,
    actions: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    legal_flat: np.ndarray,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    order = rng.permutation(len(positions))
    model.train()
    policy_total = 0.0
    value_total = 0.0
    batches = 0

    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        count = len(indices)
        x = torch.from_numpy(positions[indices].astype(np.float32, copy=False)).to(device)
        target_actions = torch.from_numpy(actions[indices]).to(device=device, dtype=torch.long)
        target_values = torch.from_numpy(values[indices]).to(device=device, dtype=torch.float32)

        starts = offsets[indices]
        ends = offsets[indices + 1]
        counts = ends - starts
        row_ids = np.repeat(np.arange(count, dtype=np.int64), counts)
        legal_ids = np.concatenate([legal_flat[a:b] for a, b in zip(starts, ends)])
        legal_mask_np = np.zeros((count, ACTION_SIZE), dtype=np.bool_)
        legal_mask_np[row_ids, legal_ids] = True
        legal_mask = torch.from_numpy(legal_mask_np).to(device)

        policy_logits, value_prediction = model(x)
        masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        policy_loss = F.cross_entropy(masked_logits, target_actions)
        value_loss = F.mse_loss(value_prediction, target_values)
        loss = policy_loss + 0.25 * value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        policy_total += float(policy_loss.detach().item())
        value_total += float(value_loss.detach().item())
        batches += 1

    model.eval()
    return policy_total / max(1, batches), value_total / max(1, batches)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain StockfishNN on strong human games in PGN format."
    )
    parser.add_argument("--pgn-dir", default="data/pgn", help="PGN file or directory (searched recursively)")
    parser.add_argument("--base-checkpoint", default=None, help="Starting checkpoint; defaults to config model_path")
    parser.add_argument("--output", default="models/human_pretrained.pt", help="Separate output checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.00003)
    parser.add_argument("--max-games", type=int, default=0, help="0 means no game-count limit")
    parser.add_argument("--max-positions", type=int, default=100000)
    parser.add_argument("--min-elo", type=int, default=0, help="Require both player ratings to meet this Elo")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the output checkpoint")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_positions < 1:
        parser.error("epochs, batch-size, and max-positions must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")

    config = load_config()
    base_rel = args.base_checkpoint or config.get("model_path", "models/latest.pt")
    base_path = Path(base_rel)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if base_path.resolve() == output_path.resolve():
        parser.error("Output must be different from the base checkpoint; models/latest.pt is protected")
    if output_path.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output_path}. Choose another --output or pass --overwrite.")

    source = Path(args.pgn_dir)
    if not source.is_absolute():
        source = ROOT / source
    files = find_pgn_files(source)
    if not files:
        raise FileNotFoundError(
            f"No .pgn or .pgn.gz files found at {source}. Add your downloaded PGN files there."
        )
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Starting checkpoint not found: {base_path}. Train the NN first or pass --base-checkpoint."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PGN files: {len(files)}")
    print(f"Starting from existing model: {base_path}")
    positions, actions, values, offsets, legal_flat, counts = collect_positions(
        files, args.max_games, args.max_positions, args.min_elo
    )
    games_seen, games_used = (int(counts[0]), int(counts[1]))
    print(
        f"Collected {len(positions):,} positions from {games_used:,} usable games "
        f"({games_seen:,} games read)."
    )
    print(f"Training device: {device} | samples: {len(positions):,}")
    print("Illegal moves are masked out of the policy loss for every position.")

    checkpoint = torch.load(base_path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        raise RuntimeError(f"{base_path} is not a StockfishNN checkpoint (missing model_state).")
    model = ChessNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        policy_loss, value_loss = train_epoch(
            model, optimizer, positions, actions, values, offsets, legal_flat,
            device, args.batch_size, rng
        )
        print(
            f"Epoch {epoch}/{args.epochs} | policy loss {policy_loss:.5f} "
            f"| value loss {value_loss:.5f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.update(
        {
            "pretraining": "human_pgn",
            "pretraining_epochs": args.epochs,
            "pretraining_positions": int(len(positions)),
            "pretraining_games": games_used,
            "pretraining_source": str(source),
            "base_checkpoint": str(base_path),
            "learning_rate": args.learning_rate,
            "architecture": "ChessNet-18plane-policy-value-v1",
        }
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata,
        },
        output_path,
    )
    print(f"Human-game pretraining checkpoint saved: {output_path}")
    print(f"Elapsed: {time.time() - started:.1f}s")
    print("Your original models/latest.pt and replay buffer were not modified.")
    print("Evaluate this checkpoint before deciding whether to promote it for normal training/play.")


if __name__ == "__main__":
    main()
