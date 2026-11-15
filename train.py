from __future__ import annotations

import argparse
import csv
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from game_archive import archive_training_game

from chess_ai import (
    ACTION_SIZE,
    ROOT,
    ChessNet,
    encode_board,
    load_config,
    move_to_index,
    resolve_stockfish_path,
    choose_move,
)


@dataclass
class Example:
    position: np.ndarray
    teacher_action: int
    value: float
    legal_actions: list[int]
    chosen_action: int = -1
    reward: float = 0.0  # individual move-quality reward
    game_reward: float = 0.0  # shared outcome/evaluation reward for the whole game


class ReplayBuffer:
    def __init__(self, capacity: int = 30000) -> None:
        self.capacity = capacity
        self.items: deque[Example] = deque(maxlen=capacity)

    def add_many(self, examples: Iterable[Example]) -> None:
        self.items.extend(examples)

    def __len__(self) -> int:
        return len(self.items)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        examples = list(self.items)
        if not examples:
            return
        positions = np.stack([item.position for item in examples]).astype(np.uint8)
        teacher = np.asarray([item.teacher_action for item in examples], dtype=np.int32)
        values = np.asarray([item.value for item in examples], dtype=np.float32)
        chosen = np.asarray([item.chosen_action for item in examples], dtype=np.int32)
        rewards = np.asarray([item.reward for item in examples], dtype=np.float32)
        game_rewards = np.asarray([item.game_reward for item in examples], dtype=np.float32)
        offsets = [0]
        legal_flat: list[int] = []
        for item in examples:
            legal_flat.extend(item.legal_actions)
            offsets.append(len(legal_flat))
        np.savez_compressed(
            path,
            positions=positions,
            teacher=teacher,
            values=values,
            chosen=chosen,
            rewards=rewards,
            game_rewards=game_rewards,
            legal_offsets=np.asarray(offsets, dtype=np.int32),
            legal_actions=np.asarray(legal_flat, dtype=np.int32),
        )

    def load(self, path: Path) -> None:
        if not path.is_file():
            return
        with np.load(path, allow_pickle=False) as data:
            positions = data["positions"]
            teacher = data["teacher"]
            values = data["values"]
            chosen = data["chosen"]
            # Older replay buffers predate evaluation-shaped rewards.
            rewards = data["rewards"] if "rewards" in data.files else np.zeros(len(positions), dtype=np.float32)
            game_rewards = data["game_rewards"] if "game_rewards" in data.files else np.zeros(len(positions), dtype=np.float32)
            offsets = data["legal_offsets"]
            legal_flat = data["legal_actions"]
            start = max(0, len(positions) - self.capacity)
            for i in range(start, len(positions)):
                legal = legal_flat[offsets[i] : offsets[i + 1]].astype(np.int32).tolist()
                self.items.append(
                    Example(
                        positions[i].copy(),
                        int(teacher[i]),
                        float(values[i]),
                        legal,
                        int(chosen[i]),
                        float(rewards[i]),
                        float(game_rewards[i]),
                    )
                )


def configure_engine(engine: chess.engine.SimpleEngine, config: dict, skill: int, threads: int, hash_mb: int) -> None:
    options = engine.options
    settings = {}
    if "Skill Level" in options:
        settings["Skill Level"] = max(0, min(20, int(skill)))
    if "Threads" in options:
        settings["Threads"] = max(1, int(threads))
    if "Hash" in options:
        settings["Hash"] = max(16, int(hash_mb))
    if settings:
        engine.configure(settings)


def side_result(result: str, color: chess.Color) -> float:
    if result == "1-0":
        white_score = 1.0
    elif result == "0-1":
        white_score = -1.0
    else:
        white_score = 0.0
    return white_score if color == chess.WHITE else -white_score


def _engine_score_cp(info: dict, board: chess.Board) -> int:
    score = info.get("score")
    if score is None:
        return 0
    pov_score = score.pov(board.turn)
    value = pov_score.score(mate_score=100000)
    return int(value if value is not None else 0)


def centipawns_to_value(cp: int | float) -> float:
    """Map side-to-move Stockfish evaluation to the value head's [-1, 1] scale."""
    return float(np.tanh(float(cp) / 600.0))


def _material_balance_cp(board: chess.Board, perspective: chess.Color) -> int:
    """Material balance in centipawns from perspective (kings excluded)."""
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }
    total = 0
    for piece_type, value in values.items():
        total += len(board.pieces(piece_type, perspective)) * value
        total -= len(board.pieces(piece_type, not perspective)) * value
    return total


def play_training_game(
    model: ChessNet,
    engine: chess.engine.SimpleEngine,
    device: torch.device,
    depth: int,
    max_plies: int,
    temperature: float,
    ai_color_override: chess.Color | None = None,
) -> tuple[list[Example], str, int, chess.Board, chess.Color, list[int]]:
    board = chess.Board()
    ai_color = ai_color_override if ai_color_override is not None else random.choice([chess.WHITE, chess.BLACK])
    pending: list[tuple[np.ndarray, int, list[int], int, chess.Color]] = []
    # Each score is Stockfish's evaluation from the side-to-move perspective
    # before that ply. The final position is evaluated once after the game.
    evaluations_stm: list[int] = []
    material_history: list[int] = []

    for _ in range(max_plies):
        # Do not auto-claim optional threefold/50-move draws. Only automatic
        # rule endings stop the game; a player would otherwise need to claim.
        if board.is_game_over(claim_draw=False):
            break
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        material_history.append(_material_balance_cp(board, ai_color))
        # Stockfish supplies its best move and its evaluation in every position.
        # This is also the NN's teacher target on the NN's turns.
        sf_result = engine.play(
            board,
            chess.engine.Limit(depth=depth),
            info=chess.engine.INFO_SCORE | chess.engine.INFO_BASIC,
        )
        teacher_move = sf_result.move
        if teacher_move is None:
            break
        score_info = sf_result.info
        if score_info.get("score") is None:
            score_info = engine.analyse(
                board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_SCORE
            )
        evaluations_stm.append(_engine_score_cp(score_info, board))
        legal_indices = [move_to_index(move) for move in legal_moves]
        if board.turn == ai_color:
            chosen_move = choose_move(model, board, device, temperature=temperature)
            chosen_index = move_to_index(chosen_move)
        else:
            chosen_move = teacher_move
            chosen_index = -1
        pending.append(
            (
                encode_board(board),
                move_to_index(teacher_move),
                legal_indices,
                chosen_index,
                board.turn,
            )
        )
        board.push(chosen_move)

    # Record the final position's material before deciding whether this was
    # a real automatic result or merely the training ply cap.
    material_history.append(_material_balance_cp(board, ai_color))
    if board.is_game_over(claim_draw=False):
        result = board.result(claim_draw=False)
        if board.is_checkmate():
            evaluations_stm.append(-100000)
        else:
            evaluations_stm.append(0)
    else:
        # Reaching max_plies is a training cutoff, not a chess draw. Keep the
        # PGN result unfinished (*); value targets still come from Stockfish.
        result = "*"
        # The engine's final position is not searched until the next ply, so
        # explicitly evaluate this last board to complete the per-move score list.
        final_info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            info=chess.engine.INFO_SCORE | chess.engine.INFO_BASIC,
        )
        evaluations_stm.append(_engine_score_cp(final_info, board))

    # One game-level signal is assigned to every NN move in this game.
    # Completed wins/losses use the actual result; draws are neutral.
    # Games stopped by the ply cap use the final Stockfish evaluation instead
    # of pretending that the unfinished game was a draw.
    if result == "*":
        final_eval_ai = evaluations_stm[-1]
        if board.turn != ai_color:
            final_eval_ai = -final_eval_ai
        game_reward = float(np.tanh(float(final_eval_ai) / 600.0))
    else:
        game_reward = side_result(result, ai_color)

    examples: list[Example] = []
    for ply, (position, teacher_action, legal, chosen, turn) in enumerate(pending):
        terminal_outcome = side_result(result, turn)
        # Dense value supervision: every position receives Stockfish's
        # side-to-move evaluation, including games stopped by the ply cap.
        sf_value = centipawns_to_value(evaluations_stm[ply])
        if result == "*":
            value_target = sf_value
        else:
            # Final result is the stronger signal; retain some positional
            # evaluation so the value head also learns position quality.
            value_target = 0.75 * terminal_outcome + 0.25 * sf_value
        value_target = max(-1.0, min(1.0, value_target))
        shaped_reward = 0.0
        move_game_reward = game_reward if turn == ai_color and chosen >= 0 else 0.0
        if turn == ai_color and chosen >= 0 and ply + 1 < len(evaluations_stm):
            # Measure the position immediately after the NN's move, before
            # Stockfish replies. The next evaluation is from the opponent's
            # side-to-move perspective, so negate it to get the NN's POV.
            # This isolates the NN move instead of mixing in the opponent's
            # reply (which made the old reward systematically more negative).
            before_cp = float(evaluations_stm[ply])
            after_for_mover_cp = -float(evaluations_stm[ply + 1])
            material_after_index = min(ply + 1, len(material_history) - 1)
            delta_cp = after_for_mover_cp - before_cp

            # Approximate centipawn loss (CPL): how far the played move's
            # resulting position fell below Stockfish's best-line evaluation.
            # The evaluation after the NN move is converted back to the NN POV.
            move_cpl = max(0.0, before_cp - after_for_mover_cp)

            # Reward low-CPL moves directly. 0 CPL -> +1. Increasing CPL
            # smoothly lowers the reward; severe mistakes become negative.
            # This is deliberately bounded and less sensitive to extreme scores.
            cpl_reward = 1.0 - 2.0 * np.tanh(move_cpl / 300.0)

            # Retain the existing material-loss safeguard. Only penalize
            # material drops not already explained/compensated by the eval.
            material_before = material_history[ply]
            material_after = material_history[material_after_index]
            material_delta = material_after - material_before
            uncompensated_loss = max(0.0, -float(material_delta) - max(0.0, delta_cp))
            material_blunder_penalty = uncompensated_loss / 500.0

            # CPL is the primary move-quality reward; material loss is an
            # additional penalty. Keep the signal bounded for stable training.
            shaped_reward = max(-1.0, min(1.0, cpl_reward - material_blunder_penalty))
        examples.append(
            Example(position, teacher_action, value_target, legal, chosen, shaped_reward, move_game_reward)
        )
    return examples, result, len(pending), board, ai_color, evaluations_stm


def build_training_arrays(buffer: ReplayBuffer) -> tuple[np.ndarray, ...]:
    """Pack replay data once per cycle to avoid repeated Python object work."""
    items = list(buffer.items)
    positions = np.stack([item.position for item in items]).astype(np.uint8, copy=False)
    teacher = np.asarray([item.teacher_action for item in items], dtype=np.int64)
    values = np.asarray([item.value for item in items], dtype=np.float32)
    chosen = np.asarray([item.chosen_action for item in items], dtype=np.int64)
    rewards = np.asarray([item.reward for item in items], dtype=np.float32)
    game_rewards = np.asarray([item.game_reward for item in items], dtype=np.float32)
    offsets = np.zeros(len(items) + 1, dtype=np.int64)
    legal_flat_parts = [np.asarray(item.legal_actions, dtype=np.int32) for item in items]
    for i, part in enumerate(legal_flat_parts):
        offsets[i + 1] = offsets[i] + len(part)
    legal_flat = np.concatenate(legal_flat_parts) if legal_flat_parts else np.empty(0, dtype=np.int32)
    return positions, teacher, values, chosen, rewards, game_rewards, offsets, legal_flat


def train_epoch(
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    arrays: tuple[np.ndarray, ...],
    device: torch.device,
    batch_size: int,
    max_samples: int,
    reward_scale: float,
) -> tuple[float, float, float]:
    """Vectorized replay sampling/masking; fewer Python and tensor allocations."""
    positions, teacher_all, values_all, chosen_all, rewards_all, game_rewards_all, offsets, legal_flat = arrays
    sample_count = min(len(positions), max_samples)
    if sample_count == 0:
        return 0.0, 0.0, 0.0
    sampled = np.random.choice(len(positions), size=sample_count, replace=False)
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    batches = 0

    for start in range(0, sample_count, batch_size):
        idx = sampled[start : start + batch_size]
        batch_len = len(idx)
        x_np = positions[idx].astype(np.float32, copy=False)
        x = torch.from_numpy(x_np).to(device)
        teacher = torch.from_numpy(teacher_all[idx]).to(device=device, dtype=torch.long)
        values_target = torch.from_numpy(values_all[idx]).to(device)
        chosen = torch.from_numpy(chosen_all[idx]).to(device=device, dtype=torch.long)
        shaped_rewards = torch.from_numpy(rewards_all[idx]).to(device)
        game_rewards = torch.from_numpy(game_rewards_all[idx]).to(device)

        starts = offsets[idx]
        ends = offsets[idx + 1]
        counts = ends - starts
        row_ids = np.repeat(np.arange(batch_len, dtype=np.int64), counts)
        legal_ids = np.concatenate([legal_flat[a:b] for a, b in zip(starts, ends)])
        legal_mask_np = np.zeros((batch_len, ACTION_SIZE), dtype=np.bool_)
        legal_mask_np[row_ids, legal_ids] = True
        legal_mask = torch.from_numpy(legal_mask_np).to(device)

        policy_logits, value_prediction = model(x)
        masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        supervised_loss = F.cross_entropy(masked_logits, teacher)
        value_loss = F.mse_loss(value_prediction, values_target)

        policy_log_probs = F.log_softmax(masked_logits, dim=1)
        selected_rows = torch.nonzero(chosen >= 0, as_tuple=False).flatten()
        if selected_rows.numel() > 0:
            chosen_log_probs = policy_log_probs[selected_rows, chosen[selected_rows]]
            # 60% of the policy reward explains whether this specific move
            # was good; 40% carries credit/blame from the overall game result.
            combined_reward = (
                0.60 * shaped_rewards[selected_rows]
                + 0.4 * game_rewards[selected_rows]
            )
            reward_target = values_target[selected_rows] + reward_scale * combined_reward
            advantages = reward_target - value_prediction[selected_rows].detach()
            reinforce_loss = -(chosen_log_probs * advantages).mean()
        else:
            reinforce_loss = torch.zeros((), device=device)

        loss = supervised_loss + 0.5 * value_loss + 0.05 * reinforce_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals += [supervised_loss.item(), value_loss.item(), reinforce_loss.item()]
        batches += 1

    model.eval()
    return tuple((totals / max(1, batches)).tolist())

def save_checkpoint(model: ChessNet, optimizer: torch.optim.Optimizer, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a chess neural network against Stockfish.")
    parser.add_argument("--cycles", type=int, default=10, help="Training cycles to run")
    parser.add_argument("--games-per-cycle", type=int, default=8, help="NN vs Stockfish games per cycle")
    parser.add_argument("--depth", type=int, default=5, help="Stockfish search depth for moves")
    parser.add_argument("--skill", type=int, default=5, help="Stockfish skill level (0-20)")
    parser.add_argument("--threads", type=int, default=None, help="Stockfish CPU threads")
    parser.add_argument("--epochs", type=int, default=2, help="Neural-network epochs per cycle")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--max-samples", type=int, default=2500, help="Replay positions sampled per epoch")
    parser.add_argument("--hash-mb", type=int, default=512, help="Stockfish hash table size in MB")
    parser.add_argument("--torch-threads", type=int, default=4, help="CPU threads for neural-network training")
    parser.add_argument("--replay-path", default="data/replay_buffer_move_game_v3.npz", help="Replay buffer for separate move and game rewards")
    parser.add_argument("--benchmark-every", type=int, default=10, help="Run a fixed benchmark every N cycles; 0 disables")
    parser.add_argument("--benchmark-games", type=int, default=6, help="Benchmark games, alternating NN colors")
    parser.add_argument("--temperature", type=float, default=1.0, help="Exploration during training games")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--reward-scale", type=float, default=None, help="Weight of Stockfish evaluation-change reward")
    parser.add_argument("--keep-avg-cpl", type=float, default=None, help="Also keep losses with NN average centipawn loss at or below this value")
    parser.add_argument("--fresh", action="store_true", help="Start new weights and an empty replay buffer")
    args = parser.parse_args()

    random.seed()
    np.random.seed()
    torch.manual_seed(random.randrange(0, 2**31 - 1))
    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass

    config = load_config()
    model_rel = Path(config.get("model_path", "models/latest.pt"))
    replay_rel = Path(args.replay_path)
    model_path = model_rel if model_rel.is_absolute() else ROOT / model_rel
    replay_path = replay_rel if replay_rel.is_absolute() else ROOT / replay_rel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    buffer = ReplayBuffer()
    previous_cycles = 0

    if not args.fresh and model_path.is_file():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        previous_cycles = int(checkpoint.get("metadata", {}).get("cycles_completed", 0))
        print(f"Resuming model from: {model_path}")
        print(f"Previous training cycles: {previous_cycles}")
    else:
        print("Starting a new model from random weights.")

    if not args.fresh:
        buffer.load(replay_path)
    print(f"Replay positions loaded: {len(buffer):,}")
    print(f"Training device: {device}")

    stockfish_path = resolve_stockfish_path(config)
    skill = args.skill
    threads = args.threads if args.threads is not None else 2
    depth = args.depth
    keep_avg_cpl = args.keep_avg_cpl if args.keep_avg_cpl is not None else float(config.get("good_game_max_avg_cpl", 60))
    reward_scale = args.reward_scale if args.reward_scale is not None else float(config.get("reward_scale", 0.25))

    print(f"Stockfish: {stockfish_path}")
    print(f"Opponent skill: {skill}/20 | depth: {depth} | games/cycle: {args.games_per_cycle}")
    print(f"Performance: Stockfish {threads} thread(s), {args.hash_mb} MB hash | PyTorch {args.torch_threads} thread(s) | batch {args.batch_size}")
    print(f"Stockfish game evaluations: enabled | value targets: Stockfish eval + game outcome")
    print(f"Replay buffer: {replay_path} | samples/epoch: {args.max_samples:,}")
    print(f"Good-game loss cutoff: {keep_avg_cpl:.0f} cp")
    print(f"Reward system: 75% individual move quality + 25% whole-game result/evaluation | reward scale: {reward_scale:.2f}")
    print("Training starts. Checkpoints are written after each cycle.\n")

    log_path = ROOT / "training_log_eval_v2.csv"
    log_exists = log_path.exists()
    with chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as engine:
        configure_engine(engine, config, skill, threads, args.hash_mb)
        for cycle in range(1, args.cycles + 1):
            cycle_start = time.time()
            cycle_examples: list[Example] = []
            wins = draws = losses = unfinished = total_positions = good_games = 0
            model.eval()
            for game_number in range(1, args.games_per_cycle + 1):
                examples, result, positions, game_board, ai_color, evaluations_stm = play_training_game(
                    model, engine, device, depth, args.max_plies, args.temperature
                )
                cycle_examples.extend(examples)
                total_positions += positions
                ai_color_result = side_result(result, ai_color)
                if result == "*":
                    unfinished += 1
                    result_label = "unfinished (ply cap)"
                elif ai_color_result > 0:
                    wins += 1
                    result_label = "win"
                elif ai_color_result < 0:
                    losses += 1
                    result_label = "loss"
                else:
                    draws += 1
                    result_label = "draw"

                game_metrics = archive_training_game(
                    game_board,
                    ai_color,
                    result,
                    evaluations_stm,
                    cycle=previous_cycles + cycle,
                    game_number=game_number,
                    analysis_depth=depth,
                    max_avg_cpl=keep_avg_cpl,
                )
                if game_metrics["good"]:
                    good_games += 1
                print(
                    f"  Game {game_number}/{args.games_per_cycle}: {result} | "
                    f"NN: {result_label} | avg NN CPL: {game_metrics['avg_ai_cpl']:.1f} | "
                    f"positions evaluated: {len(evaluations_stm)} | "
                    f"{'SAVED AS GOOD' if game_metrics['good'] else 'archived'}"
                )

            buffer.add_many(cycle_examples)
            if len(buffer) == 0:
                raise RuntimeError("No training positions were collected. Check Stockfish and try again.")

            # Pack the replay buffer once; this uses some extra RAM (~300 MB at
            # the configured capacity) to reduce Python overhead in every epoch.
            training_arrays = build_training_arrays(buffer)
            losses_avg = [0.0, 0.0, 0.0]
            for _ in range(max(1, args.epochs)):
                values = train_epoch(
                    model, optimizer, training_arrays, device, args.batch_size, max_samples=args.max_samples,
                    reward_scale=reward_scale,
                )
                losses_avg = [a + b for a, b in zip(losses_avg, values)]
            losses_avg = [value / max(1, args.epochs) for value in losses_avg]

            benchmark_wins = benchmark_draws = benchmark_losses = benchmark_unfinished = 0
            completed = previous_cycles + cycle
            benchmark_ran = args.benchmark_every > 0 and completed % args.benchmark_every == 0
            if benchmark_ran:
                print(f"  Running {args.benchmark_games}-game benchmark (fixed colors, greedy NN)...")
                model.eval()
                for benchmark_index in range(args.benchmark_games):
                    fixed_color = chess.WHITE if benchmark_index % 2 == 0 else chess.BLACK
                    _, benchmark_result, _, _, benchmark_ai_color, _ = play_training_game(
                        model, engine, device, depth, args.max_plies, 0.0,
                        ai_color_override=fixed_color,
                    )
                    score = side_result(benchmark_result, benchmark_ai_color)
                    if benchmark_result == "*":
                        benchmark_unfinished += 1
                    elif score > 0:
                        benchmark_wins += 1
                    elif score < 0:
                        benchmark_losses += 1
                    else:
                        benchmark_draws += 1
                print(
                    f"  BENCHMARK: W/D/L {benchmark_wins}/{benchmark_draws}/{benchmark_losses} "
                    f"| unfinished {benchmark_unfinished}\n"
                )

            metadata = {
                "cycles_completed": completed,
                "training_positions": len(buffer),
                "stockfish_skill": skill,
                "stockfish_depth": depth,
                "last_cycle_wins": wins,
                "last_cycle_draws": draws,
                "last_cycle_losses": losses,
                "last_cycle_unfinished": unfinished,
                "last_cycle_good_games": good_games,
                "good_game_max_avg_cpl": keep_avg_cpl,
                "reward_scale": reward_scale,
                "value_target_version": 2,
                "performance_profile": "balanced_cpu_v1",
                "torch_threads": args.torch_threads,
                "stockfish_threads": threads,
                "stockfish_hash_mb": args.hash_mb,
                "benchmark_wins": benchmark_wins,
                "benchmark_draws": benchmark_draws,
                "benchmark_losses": benchmark_losses,
                "benchmark_unfinished": benchmark_unfinished,
            }
            save_checkpoint(model, optimizer, model_path, metadata)
            save_checkpoint(
                model,
                optimizer,
                ROOT / "models" / f"cycle_{completed:04d}.pt",
                metadata,
            )
            buffer.save(replay_path)

            elapsed = time.time() - cycle_start
            ai_examples = [item for item in cycle_examples if item.chosen_action >= 0]
            ai_rewards = [item.reward for item in ai_examples]
            ai_game_rewards = [item.game_reward for item in ai_examples]
            mean_reward = float(np.mean(ai_rewards)) if ai_rewards else 0.0
            mean_game_reward = float(np.mean(ai_game_rewards)) if ai_game_rewards else 0.0
            print(
                f"Cycle {completed} saved | W/D/L {wins}/{draws}/{losses} | unfinished {unfinished} | "
                f"good games {good_games} | "
                f"positions {len(buffer):,} | policy {losses_avg[0]:.4f} "
                f"value {losses_avg[1]:.4f} | move reward {mean_reward:+.3f} | game reward {mean_game_reward:+.3f} | "
                f"{('benchmark W/D/L ' + str(benchmark_wins) + '/' + str(benchmark_draws) + '/' + str(benchmark_losses) + ' | ') if benchmark_ran else ''}"
                f"{elapsed:.1f}s\n"
            )
            write_log(log_path, log_exists, {
                "cycle": completed,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "new_positions": total_positions,
                "replay_positions": len(buffer),
                "policy_loss": f"{losses_avg[0]:.6f}",
                "value_loss": f"{losses_avg[1]:.6f}",
                "seconds": f"{elapsed:.1f}",
                "benchmark_wins": benchmark_wins if benchmark_ran else "",
                "benchmark_draws": benchmark_draws if benchmark_ran else "",
                "benchmark_losses": benchmark_losses if benchmark_ran else "",
                "benchmark_unfinished": benchmark_unfinished if benchmark_ran else "",
            })
            log_exists = True

    print("Training finished. Launch play.bat to play, or replay.bat to review archived games.")


def os_cpu_count() -> int | None:
    try:
        import os
        return os.cpu_count()
    except Exception:
        return None


def write_log(path: Path, exists: bool, row: dict[str, str | int]) -> None:
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
