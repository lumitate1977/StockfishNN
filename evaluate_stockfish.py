from __future__ import annotations

import argparse
import csv
from pathlib import Path

import chess
import chess.engine
import torch

from chess_ai import ROOT, ChessNet, load_config, resolve_stockfish_path
from train import configure_engine, play_training_game, side_result


def load_checkpoint(path: Path, device: torch.device) -> tuple[ChessNet, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        raise RuntimeError(f"Not a StockfishNN checkpoint (missing model_state): {path}")
    model = ChessNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint.get("metadata", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare StockfishNN checkpoints against the same Stockfish opponent."
    )
    parser.add_argument("--baseline", default="models/latest.pt",
                        help="Original checkpoint to compare against")
    parser.add_argument("--candidate", default="models/human_pretrained.pt",
                        help="Candidate checkpoint, e.g. the PGN-pretrained model")
    parser.add_argument("--games-per-model", type=int, default=10,
                        help="Games per checkpoint; colors alternate")
    parser.add_argument("--depth", type=int, default=None,
                        help="Stockfish search depth (defaults to config.json training_depth)")
    parser.add_argument("--skill", type=int, default=None,
                        help="Stockfish Skill Level 0-20 (defaults to config.json stockfish_skill)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Stockfish threads (defaults to config.json stockfish_threads)")
    parser.add_argument("--hash-mb", type=int, default=None,
                        help="Stockfish hash MB (defaults to config.json stockfish_hash_mb)")
    parser.add_argument("--max-plies", type=int, default=160,
                        help="Maximum plies per game; unfinished games are reported separately")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--output", default="evaluation_results.csv",
                        help="CSV output path")
    args = parser.parse_args()

    if args.games_per_model < 1 or args.max_plies < 2:
        parser.error("--games-per-model must be >= 1 and --max-plies must be >= 2")

    config = load_config()
    depth = args.depth if args.depth is not None else int(config.get("training_depth", 6))
    skill = args.skill if args.skill is not None else int(config.get("stockfish_skill", 5))
    threads = args.threads if args.threads is not None else int(config.get("stockfish_threads", 2))
    hash_mb = args.hash_mb if args.hash_mb is not None else int(config.get("stockfish_hash_mb", 256))

    def resolve_checkpoint(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    baseline_path = resolve_checkpoint(args.baseline)
    candidate_path = resolve_checkpoint(args.candidate)
    if baseline_path.resolve() == candidate_path.resolve():
        parser.error("--baseline and --candidate must be different checkpoint files")

    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stockfish depth: {depth} | skill: {skill}/20 | threads: {threads} | hash: {hash_mb} MB")
    print(f"Games per model: {args.games_per_model} | max plies: {args.max_plies}")
    print("Each checkpoint gets the same number of White and Black games as evenly as possible.")
    print("NN move selection is greedy (temperature 0); no model weights are changed.")

    stockfish_path = resolve_stockfish_path(config)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    summaries: dict[str, dict[str, int | float]] = {}

    with chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as engine:
        configure_engine(engine, config, skill, threads, hash_mb)

        for label, checkpoint_path in (("baseline", baseline_path), ("candidate", candidate_path)):
            model, metadata = load_checkpoint(checkpoint_path, device)
            print(f"\n=== {label.upper()}: {checkpoint_path.name} ===")
            if metadata.get("pretraining") == "human_pgn":
                print("Checkpoint metadata: human PGN pretraining")
                print(f"Base checkpoint: {metadata.get('base_checkpoint', 'not recorded')}")
                print(f"PGN positions: {metadata.get('pretraining_positions', 'not recorded')}")
            wins = draws = losses = unfinished = 0
            points = 0.0
            completed = 0

            for game_index in range(args.games_per_model):
                ai_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK
                color_name = "White" if ai_color == chess.WHITE else "Black"
                _, result, _, final_board, actual_color, evaluations = play_training_game(
                    model, engine, device, depth, args.max_plies, 0.0,
                    ai_color_override=ai_color,
                )
                score = side_result(result, actual_color)
                if result == "*":
                    unfinished += 1
                    final_cp = evaluations[-1] if evaluations else 0
                    if final_board.turn != actual_color:
                        final_cp = -final_cp
                    outcome = "unfinished"
                    print(f"  Game {game_index + 1}/{args.games_per_model} ({color_name}): "
                          f"unfinished at {args.max_plies} plies | final eval {final_cp / 100:+.2f}")
                else:
                    completed += 1
                    if score > 0:
                        wins += 1
                        points += 1.0
                        outcome = "win"
                    elif score < 0:
                        losses += 1
                        outcome = "loss"
                    else:
                        draws += 1
                        points += 0.5
                        outcome = "draw"
                    print(f"  Game {game_index + 1}/{args.games_per_model} ({color_name}): "
                          f"{result} | NN {outcome}")

                rows.append({
                    "checkpoint": label,
                    "checkpoint_path": str(checkpoint_path),
                    "game": game_index + 1,
                    "nn_color": color_name,
                    "result": result,
                    "outcome": outcome,
                    "stockfish_depth": depth,
                    "stockfish_skill": skill,
                    "max_plies": args.max_plies,
                })

            score_pct = (points / completed * 100.0) if completed else 0.0
            summaries[label] = {
                "wins": wins, "draws": draws, "losses": losses,
                "unfinished": unfinished, "completed": completed,
                "score_pct": score_pct,
            }
            print(f"  SUMMARY: W/D/L {wins}/{draws}/{losses} | unfinished {unfinished} "
                  f"| score {score_pct:.1f}% of completed games")
            del model

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetailed results saved to: {output_path}")
    print("\nCOMPARISON (completed games only; unfinished games excluded from score):")
    for label, summary in summaries.items():
        print(f"  {label}: {summary['score_pct']:.1f}% | "
              f"W/D/L {summary['wins']}/{summary['draws']}/{summary['losses']} "
              f"| unfinished {summary['unfinished']}")


if __name__ == "__main__":
    main()
