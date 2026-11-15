from __future__ import annotations

import argparse
import re
from pathlib import Path

import chess.pgn

from game_archive import ALL_GAMES_PATH, GOOD_GAMES_PATH, read_games


def safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned[:70] or fallback


def export_games(source: Path, output: Path, *, split: bool = False, overwrite: bool = False) -> list[Path]:
    games = read_games(source)
    if not games:
        raise RuntimeError(f"No PGN games found in {source}")

    output_paths: list[Path] = []
    if split:
        output.mkdir(parents=True, exist_ok=True)
        for index, game in enumerate(games, start=1):
            headers = game.headers
            white = safe_name(headers.get("White", "White"), "White")
            black = safe_name(headers.get("Black", "Black"), "Black")
            result = safe_name(headers.get("Result", "unknown"), "unknown")
            path = output / f"{index:05d}_{white}_vs_{black}_{result}.pgn"
            if path.exists() and not overwrite:
                raise FileExistsError(f"Export would overwrite {path}. Choose another folder or pass --overwrite.")
            exporter = chess.pgn.FileExporter
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                game.accept(exporter(handle))
                handle.write("\n")
            output_paths.append(path)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not overwrite:
            raise FileExistsError(f"Export would overwrite {output}. Choose another filename or pass --overwrite.")
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for game in games:
                game.accept(chess.pgn.FileExporter(handle))
                handle.write("\n\n")
        output_paths.append(output)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export StockfishNN archived games as standard PGN files.")
    parser.add_argument("--source", default=str(ALL_GAMES_PATH), help="Source archive (.pgn)")
    parser.add_argument("--output", default="exports/games.pgn", help="Output .pgn file, or folder with --split")
    parser.add_argument("--split", action="store_true", help="Write each archived game to its own PGN file")
    parser.add_argument("--good-games", action="store_true", help="Use games/good_games.pgn as the source")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files")
    args = parser.parse_args()
    source = GOOD_GAMES_PATH if args.good_games else Path(args.source).expanduser()
    output = Path(args.output).expanduser()
    if not source.is_absolute():
        source = Path(__file__).resolve().parent / source
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    paths = export_games(source, output, split=args.split, overwrite=args.overwrite)
    print(f"Exported {len(paths)} PGN file(s) from {source}")
    for path in paths[:10]:
        print(f"  {path}")
    if len(paths) > 10:
        print(f"  ... and {len(paths) - 10} more")


if __name__ == "__main__":
    main()
