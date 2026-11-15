from __future__ import annotations

import argparse
import math
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

from chess_ai import ROOT, ChessNet, choose_move, encode_board, load_config, move_to_index, resolve_stockfish_path
from game_archive import ALL_GAMES_PATH, append_game
from train import configure_engine, side_result


def load_net(path_text: str, device: torch.device):
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ChessNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, path


def eval_cp(info: dict, board: chess.Board) -> int:
    score = info["score"].pov(board.turn).score(mate_score=100000)
    return int(score if score is not None else 0)


def optimize_game(model, optimizer, records, outcome: float, device):
    if not records:
        return 0.0
    model.train()
    losses = []
    for board, move, cpl in records:
        x = torch.from_numpy(encode_board(board).astype(np.float32)).unsqueeze(0).to(device)
        logits, value = model(x)
        legal = list(board.legal_moves)
        legal_ids = torch.tensor([move_to_index(m) for m in legal], dtype=torch.long, device=device)
        masked = logits[0, legal_ids]
        chosen_index = next(i for i, candidate in enumerate(legal) if candidate == move)
        log_prob = F.log_softmax(masked, dim=0)[chosen_index]
        move_quality = 1.0 - 2.0 * math.tanh(max(0, cpl) / 300.0)
        reward = 0.70 * outcome + 0.30 * move_quality
        target = torch.tensor([outcome], dtype=torch.float32, device=device)
        losses.append(-log_prob * reward + 0.20 * F.mse_loss(value.reshape(1), target))
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()
    return float(loss.detach().item())


def save_trained(path: Path, model, optimizer, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "metadata": metadata}, path)


def main():
    parser = argparse.ArgumentParser(description="Play, evaluate, or train StockfishNN models against each other or Stockfish.")
    parser.add_argument("--white", default="models/latest.pt", help="White checkpoint path, or 'stockfish'")
    parser.add_argument("--black", default="models/human_pretrained.pt", help="Black checkpoint path, or 'stockfish'")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--skill", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--hash-mb", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--train", action="store_true", help="Apply game-result and Stockfish CPL policy-gradient updates")
    parser.add_argument("--white-output", default="models/arena_white.pt")
    parser.add_argument("--black-output", default="models/arena_black.pt")
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--overwrite-output", action="store_true", help="Allow replacing existing arena output checkpoints")
    parser.add_argument("--archive", default=str(ALL_GAMES_PATH.relative_to(ROOT)))
    args = parser.parse_args()

    if args.games < 1 or args.max_plies < 2:
        parser.error("--games must be >= 1 and --max-plies must be >= 2")
    white_sf, black_sf = args.white.lower() == "stockfish", args.black.lower() == "stockfish"
    if white_sf and black_sf:
        parser.error("At least one side must be a neural-network checkpoint.")
    if args.train and not (white_sf or black_sf) and Path(args.white).resolve() == Path(args.black).resolve():
        parser.error("Training both sides requires two distinct checkpoint files.")
    if args.train:
        output_paths = []
        for output in (args.white_output, args.black_output):
            path = Path(output)
            if not path.is_absolute():
                path = ROOT / path
            output_paths.append(path.resolve())
        source_paths = {Path(args.white).resolve(), Path(args.black).resolve()}
        if len(set(output_paths)) != len(output_paths):
            parser.error("White and Black training outputs must be different files.")
        if any(path in source_paths for path in output_paths):
            parser.error("Arena outputs must not overwrite either source checkpoint.")
        existing = [path for path in output_paths if path.exists()]
        if existing and not args.overwrite_output:
            parser.error("Arena output already exists: " + ", ".join(map(str, existing)) +
                         ". Choose new --white-output/--black-output paths or pass --overwrite-output.")

    config = load_config()
    depth = args.depth if args.depth is not None else int(config.get("training_depth", 6))
    skill = args.skill if args.skill is not None else int(config.get("stockfish_skill", 5))
    threads = args.threads if args.threads is not None else int(config.get("stockfish_threads", 2))
    hash_mb = args.hash_mb if args.hash_mb is not None else int(config.get("stockfish_hash_mb", 256))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models, checkpoints, paths, optimizers = {}, {}, {}, {}
    for color, is_sf, name in ((chess.WHITE, white_sf, args.white), (chess.BLACK, black_sf, args.black)):
        if is_sf:
            continue
        model, checkpoint, path = load_net(name, device)
        models[color], checkpoints[color], paths[color] = model, checkpoint, path
        optimizers[color] = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        if args.train and "optimizer_state" in checkpoint:
            # New arena optimizer uses the requested small learning rate; do not import
            # optimizer moments from a different training objective.
            pass

    stockfish_path = resolve_stockfish_path(config)
    archive_path = Path(args.archive)
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {chess.WHITE: {"w": 0, "d": 0, "l": 0}, chess.BLACK: {"w": 0, "d": 0, "l": 0}}

    print(f"Device: {device} | Stockfish analysis depth: {depth} | games: {args.games}")
    print(f"White: {'Stockfish' if white_sf else paths[chess.WHITE].name}")
    print(f"Black: {'Stockfish' if black_sf else paths[chess.BLACK].name}")
    print("Stockfish analyzes every position; CPL compares the best root evaluation with a search restricted to the move actually played.")
    if args.train:
        print("TRAINING ENABLED: source checkpoints are not overwritten; updated copies go to the output paths.")

    with chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as engine:
        configure_engine(engine, config, skill, threads, hash_mb)
        for game_number in range(1, args.games + 1):
            board = chess.Board()
            evaluations_white: list[int] = []
            move_records = {chess.WHITE: [], chess.BLACK: []}
            pgn = chess.pgn.Game()
            pgn.headers["Event"] = "StockfishNN Arena"
            pgn.headers["Site"] = "StockfishNN"
            pgn.headers["Round"] = str(game_number)
            pgn.headers["White"] = "Stockfish" if white_sf else paths[chess.WHITE].stem
            pgn.headers["Black"] = "Stockfish" if black_sf else paths[chess.BLACK].stem
            pgn.headers["AnalysisDepth"] = str(depth)
            node = pgn
            cpls = {chess.WHITE: [], chess.BLACK: []}

            while not board.is_game_over(claim_draw=False) and len(board.move_stack) < args.max_plies:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                best_cp = eval_cp(info, board)
                if not evaluations_white:
                    pgn.comment = f"[%eval {best_cp / 100:+.2f}] Starting position"
                mover = board.turn
                if mover in models:
                    move = choose_move(models[mover], board, device, temperature=0.0)
                else:
                    pv = info.get("pv", [])
                    move = pv[0] if pv else engine.play(board, chess.engine.Limit(depth=depth)).move
                if move not in board.legal_moves:
                    raise RuntimeError(f"Illegal move returned for {'White' if mover else 'Black'}: {move}")
                # Same-root comparison: best evaluation versus the played move.
                played_info = engine.analyse(
                    board, chess.engine.Limit(depth=depth), root_moves=[move]
                )
                played_cp = eval_cp(played_info, board)
                cpl = max(0, min(20000, best_cp - played_cp))
                evaluations_white.append(best_cp if mover == chess.WHITE else -best_cp)
                snapshot = board.copy(stack=False)
                board.push(move)
                cpls[mover].append(cpl)
                if mover in models:
                    move_records[mover].append((snapshot, move, cpl))
                node = node.add_variation(move)
                white_eval_after = played_cp if mover == chess.WHITE else -played_cp
                actor = "Stockfish" if mover not in models else paths[mover].stem
                node.comment = (
                    f"[%eval {white_eval_after / 100:+.2f}] {actor} centipawn loss: {cpl} cp "
                    f"(analysis depth {depth})"
                )

            if board.is_game_over(claim_draw=False):
                result = board.result(claim_draw=False)
                termination = board.outcome(claim_draw=False).termination.name
            else:
                result = "*"
                termination = f"Move limit ({args.max_plies} plies)"
            if result == "*":
                final_info = engine.analyse(board, chess.engine.Limit(depth=depth))
                final_cp = eval_cp(final_info, board)
                evaluations_white.append(final_cp if board.turn else -final_cp)
            else:
                evaluations_white.append(0 if board.is_checkmate() else 0)

            pgn.headers["Result"] = result
            pgn.headers["Termination"] = termination
            for color, label in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
                avg = sum(cpls[color]) / len(cpls[color]) if cpls[color] else 0.0
                pgn.headers[f"Avg{label}CPL"] = f"{avg:.1f}"
                pgn.headers[f"Max{label}CPL"] = str(max(cpls[color], default=0))
                pgn.headers[f"{label}Moves"] = str(len(cpls[color]))
            pgn.headers["AvgCPL"] = f"{(sum(cpls[chess.WHITE]) + sum(cpls[chess.BLACK])) / max(1, len(cpls[chess.WHITE]) + len(cpls[chess.BLACK])):.1f}"
            append_game(pgn, archive_path)

            print(f"Game {game_number}/{args.games}: {pgn.headers['White']} vs {pgn.headers['Black']} | {result} | "
                  f"avg CPL W/B {pgn.headers['AvgWhiteCPL']}/{pgn.headers['AvgBlackCPL']}")

            if args.train:
                for color in (chess.WHITE, chess.BLACK):
                    if color not in models:
                        continue
                    score = side_result(result, color)
                    if result == "*":
                        score = math.tanh((evaluations_white[-1] if color == chess.WHITE else -evaluations_white[-1]) / 600.0)
                    optimize_game(models[color], optimizers[color], move_records[color], score, device)
                for color, output in ((chess.WHITE, args.white_output), (chess.BLACK, args.black_output)):
                    if color not in models:
                        continue
                    output_path = Path(output)
                    if not output_path.is_absolute():
                        output_path = ROOT / output_path
                    save_trained(output_path, models[color], optimizers[color], {
                        "training_mode": "arena_policy_gradient",
                        "opponent": "Stockfish" if (black_sf if color == chess.WHITE else white_sf) else "neural_network",
                        "games_trained": game_number,
                        "source_checkpoint": str(paths[color]),
                        "stockfish_depth": depth,
                        "stockfish_skill": skill,
                    })

            for color in (chess.WHITE, chess.BLACK):
                score = side_result(result, color)
                if result == "*":
                    continue
                key = "w" if score > 0 else "l" if score < 0 else "d"
                totals[color][key] += 1

    print("\nCompleted-game results:")
    for color, label, is_sf in ((chess.WHITE, "White", white_sf), (chess.BLACK, "Black", black_sf)):
        if is_sf:
            continue
        t = totals[color]
        print(f"  {label} {paths[color].stem}: W/D/L {t['w']}/{t['d']}/{t['l']}")
    print(f"Archived PGN games: {archive_path}")
    if args.train:
        print("Arena training copies saved. Original input checkpoints were left untouched.")


if __name__ == "__main__":
    main()
