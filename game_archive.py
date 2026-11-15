from __future__ import annotations

import datetime as _datetime
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parent
GAMES_DIR = ROOT / "games"
ALL_GAMES_PATH = GAMES_DIR / "all_games.pgn"
GOOD_GAMES_PATH = GAMES_DIR / "good_games.pgn"


def _format_eval(centipawns: int) -> str:
    value = centipawns / 100.0
    return f"{value:+.2f}"


def _game_nodes(game: chess.pgn.Game) -> list[chess.pgn.GameNode]:
    nodes: list[chess.pgn.GameNode] = [game]
    node: chess.pgn.GameNode = game
    while node.variations:
        node = node.variations[0]
        nodes.append(node)
    return nodes


def build_annotated_game(
    final_board: chess.Board,
    ai_color: chess.Color,
    result: str,
    evals_stm: list[int],
    *,
    cycle: int,
    game_number: int,
    analysis_depth: int,
    ai_name: str = "StockfishNN",
) -> tuple[chess.pgn.Game, dict[str, float | int | bool]]:
    """Build a PGN with Stockfish evaluations and per-move centipawn loss.

    evals_stm[i] is the evaluation from the point of view of the side to move
    in position i. There must be one more evaluation than there are moves.
    """
    moves = list(final_board.move_stack)
    if len(evals_stm) < len(moves) + 1:
        raise ValueError("Game analysis is incomplete: expected an evaluation for every position.")

    game = chess.pgn.Game.from_board(final_board)
    game.headers["Event"] = "StockfishNN Training Game"
    game.headers["Site"] = "StockfishNN"
    game.headers["Date"] = _datetime.date.today().strftime("%Y.%m.%d")
    game.headers["Round"] = f"{cycle}.{game_number}"
    game.headers["White"] = ai_name if ai_color == chess.WHITE else "Stockfish"
    game.headers["Black"] = ai_name if ai_color == chess.BLACK else "Stockfish"
    game.headers["Result"] = result
    game.headers["AIColor"] = "White" if ai_color == chess.WHITE else "Black"
    game.headers["AIGameResult"] = _result_for_color(result, ai_color)
    game.headers["TrainingCycle"] = str(cycle)
    game.headers["GameNumber"] = str(game_number)
    game.headers["StockfishDepth"] = str(analysis_depth)

    position = final_board.root()
    move_cpls: list[int] = []
    ai_cpls: list[int] = []
    stockfish_cpls: list[int] = []
    white_evals: list[int] = []

    for ply in range(len(moves) + 1):
        stm_eval = int(evals_stm[ply])
        white_evals.append(stm_eval if position.turn == chess.WHITE else -stm_eval)
        if ply < len(moves):
            before_for_mover = int(evals_stm[ply])
            after_for_mover = -int(evals_stm[ply + 1])
            cpl = max(0, min(20000, before_for_mover - after_for_mover))
            move_cpls.append(cpl)
            if position.turn == ai_color:
                ai_cpls.append(cpl)
            else:
                stockfish_cpls.append(cpl)
            position.push(moves[ply])

    avg_ai_cpl = float(sum(ai_cpls) / len(ai_cpls)) if ai_cpls else 0.0
    max_ai_cpl = max(ai_cpls, default=0)
    game.headers["AIMoves"] = str(len(ai_cpls))
    game.headers["AvgAICPL"] = f"{avg_ai_cpl:.1f}"
    game.headers["MaxAICPL"] = str(max_ai_cpl)
    game.headers["AvgStockfishCPL"] = (
        f"{sum(stockfish_cpls) / len(stockfish_cpls):.1f}" if stockfish_cpls else "0.0"
    )

    nodes = _game_nodes(game)
    if nodes:
        nodes[0].comment = f"[%eval {_format_eval(white_evals[0])}] Stockfish starting evaluation"
    for ply, move in enumerate(moves):
        node = nodes[ply + 1]
        # Move color alternates from the starting side to move.
        mover_color = final_board.root().turn if ply % 2 == 0 else not final_board.root().turn
        actor = "NN" if mover_color == ai_color else "Stockfish"
        node.comment = (
            f"[%eval {_format_eval(white_evals[ply + 1])}] "
            f"{actor} centipawn loss: {move_cpls[ply]} cp"
        )

    ai_result = _result_for_color(result, ai_color)
    # A win/draw is always kept. Strong losses are also retained using a
    # configurable average centipawn-loss cutoff in should_keep_game().
    game.headers["GoodGame"] = "false"
    metrics: dict[str, float | int | bool] = {
        "avg_ai_cpl": avg_ai_cpl,
        "max_ai_cpl": max_ai_cpl,
        "ai_moves": len(ai_cpls),
        "ai_result": ai_result,
        "good": False,
    }
    return game, metrics


def _result_for_color(result: str, color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result not in ("1-0", "0-1"):
        return "unknown"
    white_won = result == "1-0"
    ai_won = white_won if color == chess.WHITE else not white_won
    return "win" if ai_won else "loss"


def should_keep_game(metrics: dict[str, float | int | bool], max_avg_cpl: float) -> bool:
    ai_result = str(metrics.get("ai_result", "unknown"))
    avg_ai_cpl = float(metrics.get("avg_ai_cpl", 99999.0))
    ai_moves = int(metrics.get("ai_moves", 0))
    # Keep all wins and draws; also keep losses where the NN played accurately.
    return ai_result in ("win", "draw") or (ai_moves >= 4 and avg_ai_cpl <= max_avg_cpl)


def append_game(game: chess.pgn.Game, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    rendered = game.accept(exporter)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        if path.stat().st_size:
            file.write("\n\n")
        file.write(rendered)
        file.write("\n")


def archive_training_game(
    final_board: chess.Board,
    ai_color: chess.Color,
    result: str,
    evals_stm: list[int],
    *,
    cycle: int,
    game_number: int,
    analysis_depth: int,
    max_avg_cpl: float,
) -> dict[str, float | int | bool]:
    game, metrics = build_annotated_game(
        final_board,
        ai_color,
        result,
        evals_stm,
        cycle=cycle,
        game_number=game_number,
        analysis_depth=analysis_depth,
    )
    keep = should_keep_game(metrics, max_avg_cpl)
    game.headers["GoodGame"] = "true" if keep else "false"
    metrics["good"] = keep
    append_game(game, ALL_GAMES_PATH)
    if keep:
        append_game(game, GOOD_GAMES_PATH)
    return metrics


def read_games(path: Path = ALL_GAMES_PATH) -> list[chess.pgn.Game]:
    if not path.is_file():
        return []
    games: list[chess.pgn.Game] = []
    with path.open("r", encoding="utf-8-sig") as file:
        while game := chess.pgn.read_game(file):
            games.append(game)
    return games

