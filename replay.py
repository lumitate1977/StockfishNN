from __future__ import annotations

import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import chess
import chess.pgn

from game_archive import ALL_GAMES_PATH, GOOD_GAMES_PATH, read_games


class ReplayWindow:
    # A 496px board leaves room for all replay controls in a normal Windows window.
    SQUARE = 56
    BOARD_SIZE = SQUARE * 8
    LIGHT = "#f0d9b5"
    DARK = "#b58863"
    LAST = "#cdd26a"
    PIECES = {
        chess.WHITE: {
            chess.KING: "♔", chess.QUEEN: "♕", chess.ROOK: "♖",
            chess.BISHOP: "♗", chess.KNIGHT: "♘", chess.PAWN: "♙",
        },
        chess.BLACK: {
            chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
            chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟",
        },
    }

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("StockfishNN — Game Replay & Stockfish Evaluations")
        self.root.geometry("1100x760")
        self.root.minsize(980, 700)
        self.root.configure(bg="#20242b")

        self.games: list[chess.pgn.Game] = []
        self.selected_game: chess.pgn.Game | None = None
        self.positions: list[chess.Board] = []
        self.nodes: list[chess.pgn.GameNode] = []
        self.san_moves: list[str] = []
        self.position_index = 0
        self.autoplay_job: str | None = None
        self._updating_seek = False

        self._build_ui()
        self.refresh_games()

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#20242b", padx=14, pady=12)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Game Replay",
            font=("Segoe UI", 19, "bold"),
            fg="#f4f4f5",
            bg="#20242b",
        ).pack(side="left")
        tk.Label(
            header,
            text="Saved StockfishNN training matches · Stockfish evals on every move",
            font=("Segoe UI", 10),
            fg="#aab6c5",
            bg="#20242b",
        ).pack(side="left", padx=16)

        body = tk.Frame(self.root, bg="#20242b", padx=14, pady=6)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="#20242b", width=315)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)

        tk.Label(left, text="Game archive", fg="#f4f4f5", bg="#20242b", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))
        self.filter_var = tk.StringVar(value="All games")
        filter_menu = ttk.Combobox(left, textvariable=self.filter_var, values=["All games", "Good games only"], state="readonly")
        filter_menu.pack(fill="x", pady=(0, 6))
        filter_menu.bind("<<ComboboxSelected>>", lambda _event: self.refresh_games())
        self.sort_var = tk.StringVar(value="Newest first")
        sort_menu = ttk.Combobox(
            left, textvariable=self.sort_var,
            values=["Newest first", "Best NN games"], state="readonly",
        )
        sort_menu.pack(fill="x", pady=(0, 8))
        sort_menu.bind("<<ComboboxSelected>>", lambda _event: self.refresh_games())

        list_frame = tk.Frame(left, bg="#20242b")
        list_frame.pack(fill="both", expand=True)
        self.game_list = tk.Listbox(
            list_frame,
            bg="#2b313b",
            fg="#f4f4f5",
            selectbackground="#52677f",
            selectforeground="#ffffff",
            activestyle="none",
            font=("Segoe UI", 10),
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.game_list.yview)
        self.game_list.configure(yscrollcommand=scrollbar.set)
        self.game_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.game_list.bind("<<ListboxSelect>>", self.on_select_game)

        ttk.Button(left, text="Refresh archive", command=self.refresh_games).pack(fill="x", pady=(8, 0))
        ttk.Button(left, text="Delete selected game", command=self.delete_selected_game).pack(fill="x", pady=(6, 0))
        tk.Label(
            left,
            text="All games are saved in games/all_games.pgn.\nStrong games are copied to games/good_games.pgn.",
            fg="#94a3b8",
            bg="#20242b",
            justify="left",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(12, 0))

        right = tk.Frame(body, bg="#20242b")
        right.pack(side="left", fill="both", expand=True)
        self.title_label = tk.Label(right, text="Select a game to replay", fg="#f4f4f5", bg="#20242b", font=("Segoe UI", 14, "bold"), anchor="w")
        self.title_label.pack(fill="x", pady=(0, 5))
        self.details_label = tk.Label(right, text="", fg="#cbd5e1", bg="#20242b", font=("Segoe UI", 10), anchor="w")
        self.details_label.pack(fill="x", pady=(0, 8))

        # Keep navigation ABOVE the board so it is visible without scrolling.
        controls = tk.Frame(right, bg="#20242b")
        controls.pack(fill="x", pady=(0, 5))
        ttk.Button(controls, text="|◀ Start", command=self.go_start).pack(side="left")
        ttk.Button(controls, text="◀ Previous", command=self.previous_move).pack(side="left", padx=(5, 0))
        ttk.Button(controls, text="Next ▶", command=self.next_move).pack(side="left", padx=(5, 0))
        ttk.Button(controls, text="End ▶|", command=self.go_end).pack(side="left", padx=(5, 0))
        self.autoplay_button = ttk.Button(controls, text="▶ Auto-play", command=self.toggle_autoplay)
        self.autoplay_button.pack(side="left", padx=(12, 0))
        self.speed_var = tk.StringVar(value="1.0 sec")
        ttk.Combobox(controls, textvariable=self.speed_var, values=["0.4 sec", "0.7 sec", "1.0 sec", "1.5 sec", "2.0 sec"], width=8, state="readonly").pack(side="left", padx=6)

        seek_row = tk.Frame(right, bg="#20242b")
        seek_row.pack(fill="x", pady=(0, 5))
        tk.Label(seek_row, text="Jump to move:", fg="#cbd5e1", bg="#20242b", font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))
        self.seek_scale = ttk.Scale(seek_row, from_=0, to=0, orient="horizontal", command=self.on_seek)
        self.seek_scale.pack(side="left", fill="x", expand=True)
        self.seek_text = tk.Label(seek_row, text="0 / 0", fg="#cbd5e1", bg="#20242b", font=("Segoe UI", 9), width=9)
        self.seek_text.pack(side="left", padx=(8, 0))

        self.canvas = tk.Canvas(right, width=self.BOARD_SIZE, height=self.BOARD_SIZE, highlightthickness=0, bg="#20242b")
        self.canvas.pack(anchor="center")

        self.move_label = tk.Label(right, text="", fg="#e2e8f0", bg="#20242b", font=("Segoe UI", 10), anchor="w")
        self.move_label.pack(fill="x", pady=(5, 2))
        self.eval_label = tk.Label(right, text="Stockfish evaluation: —", fg="#93c5fd", bg="#20242b", font=("Segoe UI", 10, "bold"), anchor="w")
        self.eval_label.pack(fill="x", pady=(0, 2))

        self.status_label = tk.Label(right, text="", fg="#94a3b8", bg="#20242b", font=("Segoe UI", 9), anchor="w")
        self.status_label.pack(fill="x", pady=(3, 0))

        # Keyboard shortcuts: Left/Right = one ply, Home/End = game boundaries, Space = autoplay.
        self.root.bind("<Left>", lambda _event: self.previous_move())
        self.root.bind("<Right>", lambda _event: self.next_move())
        self.root.bind("<Home>", lambda _event: self.go_start())
        self.root.bind("<End>", lambda _event: self.go_end())
        self.root.bind("<space>", lambda _event: self.toggle_autoplay())

    def refresh_games(self) -> None:
        self.stop_autoplay()
        good_only = self.filter_var.get() == "Good games only"
        archive_path = GOOD_GAMES_PATH if good_only else ALL_GAMES_PATH
        self.games = read_games(archive_path)
        self.display_games = list(reversed(self.games))
        if self.sort_var.get() == "Best NN games":
            # Rank actual NN performance: wins first, then draws, then losses.
            # Within each result group, lower average centipawn loss is better.
            result_rank = {"win": 0, "draw": 1, "loss": 2, "unknown": 3}

            def nn_game_score(game):
                headers = game.headers
                result = headers.get("AIGameResult", "unknown").strip().lower()
                try:
                    average_loss = float(headers.get("AvgAICPL", ""))
                except (TypeError, ValueError):
                    average_loss = float("inf")
                return (result_rank.get(result, 3), average_loss)

            self.display_games.sort(key=nn_game_score)
        self.game_list.delete(0, tk.END)
        for game in self.display_games:
            headers = game.headers
            result = headers.get("Result", "*")
            ai_result = headers.get("AIGameResult", "?")
            ai_color = headers.get("AIColor", "?")
            avg_cpl = headers.get("AvgAICPL", headers.get("AvgCPL", "—"))
            cycle = headers.get("TrainingCycle", headers.get("Round", "?"))
            quality = " ★" if headers.get("GoodGame", "false").lower() == "true" else ""
            label = (
                f"{headers.get('White', 'White')} vs {headers.get('Black', 'Black')} | "
                f"{result} | avg CPL {avg_cpl}{quality}"
            )
            self.game_list.insert(tk.END, label)
        if self.games:
            self.status_label.config(text=f"Loaded {len(self.games)} game(s) from {archive_path.relative_to(archive_path.parent.parent)}")
            self.game_list.selection_clear(0, tk.END)
            self.game_list.selection_set(0)
            self.game_list.event_generate("<<ListboxSelect>>")
        else:
            self.clear_board()
            self.title_label.config(text="No saved games yet")
            self.details_label.config(text="Start training. Completed NN vs Stockfish games will appear here.")
            self.move_label.config(text="")
            self.eval_label.config(text="Stockfish evaluation: —")
            self.status_label.config(text=f"Nothing found at {archive_path}")

    def on_select_game(self, _event=None) -> None:
        selection = self.game_list.curselection()
        if not selection:
            return
        if not 0 <= selection[0] < len(self.display_games):
            return
        self.load_game(self.display_games[selection[0]])

    @staticmethod
    def _game_signature(game: chess.pgn.Game) -> tuple:
        """Stable identity for a saved game across the all/good PGN archives."""
        headers = game.headers
        moves = tuple(move.uci() for move in game.mainline_moves())
        return (
            headers.get("Date", ""), headers.get("Round", ""),
            headers.get("TrainingCycle", ""), headers.get("GameNumber", ""),
            headers.get("AIColor", ""), moves,
        )

    @staticmethod
    def _write_games(path: Path, games: list[chess.pgn.Game]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            exporter = chess.pgn.FileExporter(handle)
            for game in games:
                game.accept(exporter)
                handle.write("\n\n")

    def delete_selected_game(self) -> None:
        selection = self.game_list.curselection()
        if not selection or not self.games:
            messagebox.showinfo("Delete game", "Select a game first.", parent=self.root)
            return
        if not 0 <= selection[0] < len(self.display_games):
            return
        selected = self.display_games[selection[0]]
        signature = self._game_signature(selected)
        if not messagebox.askyesno(
            "Delete archived game",
            "Permanently delete this game from the archive?\n\n"
            "If it is also in Good Games, that copy will be deleted too.",
            parent=self.root,
        ):
            return

        for path in (ALL_GAMES_PATH, GOOD_GAMES_PATH):
            existing = read_games(path)
            kept = [game for game in existing if self._game_signature(game) != signature]
            if len(kept) != len(existing):
                self._write_games(path, kept)

        self.selected_game = None
        self.positions = []
        self.nodes = []
        self.san_moves = []
        self.clear_board()
        self.refresh_games()

    def load_game(self, game: chess.pgn.Game) -> None:
        self.stop_autoplay()
        self.selected_game = game
        self.nodes = [game]
        node: chess.pgn.GameNode = game
        while node.variations:
            node = node.variations[0]
            self.nodes.append(node)

        start_board = game.board()
        self.positions = [start_board.copy()]
        self.san_moves = []
        board = start_board.copy(stack=False)
        for move in game.mainline_moves():
            self.san_moves.append(board.san(move))
            board.push(move)
            self.positions.append(board.copy())

        headers = game.headers
        white = headers.get("White", "White")
        black = headers.get("Black", "Black")
        result = headers.get("Result", "*")
        avg_cpl = headers.get("AvgAICPL", headers.get("AvgCPL", "—"))
        result_display = "Unfinished (training move limit)" if result == "*" else result
        self.title_label.config(text=f"{white} vs {black}  ·  {result_display}")
        self.details_label.config(
            text=(
                f"Round {headers.get('Round', headers.get('GameNumber', '?'))}  ·  "
                f"White avg CPL: {headers.get('AvgWhiteCPL', '—')}  ·  "
                f"Black avg CPL: {headers.get('AvgBlackCPL', '—')}  ·  "
                f"Average CPL: {avg_cpl}  ·  "
                f"{headers.get('Termination', 'Termination: not recorded')}"
            )
        )
        self.position_index = 0
        self.seek_scale.configure(to=max(0, len(self.positions) - 1))
        self.update_position()

    def clear_board(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(self.BOARD_SIZE // 2, self.BOARD_SIZE // 2, text="No games", fill="#94a3b8", font=("Segoe UI", 20))

    def draw_board(self, board: chess.Board) -> None:
        self.canvas.delete("all")
        last_move = board.peek() if board.move_stack else None
        for square in chess.SQUARES:
            file_index = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            x0, y0 = file_index * self.SQUARE, row * self.SQUARE
            x1, y1 = x0 + self.SQUARE, y0 + self.SQUARE
            color = self.LIGHT if (chess.square_file(square) + chess.square_rank(square)) % 2 else self.DARK
            if last_move and square in (last_move.from_square, last_move.to_square):
                color = self.LAST
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)
            piece = board.piece_at(square)
            if piece:
                is_black_pawn = piece.color == chess.BLACK and piece.piece_type == chess.PAWN
                self.canvas.create_text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2 - (3 if is_black_pawn else 0),
                    text=self.PIECES[piece.color][piece.piece_type],
                    font=("Segoe UI Symbol", 38 if is_black_pawn else 42),
                    fill="#fffdf4" if piece.color == chess.WHITE else "#171717",
                )
        for file_index in range(8):
            self.canvas.create_text(file_index * self.SQUARE + self.SQUARE - 9, 8 * self.SQUARE - 12,
                                    text=chess.FILE_NAMES[file_index], fill="#333333", font=("Segoe UI", 9, "bold"))
        for row in range(8):
            self.canvas.create_text(8, row * self.SQUARE + 9, text=str(8 - row), fill="#333333", font=("Segoe UI", 9, "bold"))

    def update_position(self) -> None:
        if not self.positions:
            return
        board = self.positions[self.position_index]
        self.draw_board(board)
        total = len(self.positions) - 1
        self.seek_text.config(text=f"{self.position_index} / {total}")
        self._updating_seek = True
        self.seek_scale.set(self.position_index)
        self._updating_seek = False
        if self.position_index == 0:
            move_text = "Starting position"
        else:
            move_no = (self.position_index + 1) // 2
            prefix = f"{move_no}." if self.position_index % 2 == 1 else f"{move_no}..."
            move_text = f"{prefix} {self.san_moves[self.position_index - 1]}"
        self.move_label.config(text=f"Position {self.position_index}/{total}  ·  {move_text}")

        node = self.nodes[self.position_index]
        eval_match = re.search(r"\[%eval\s+([+-]?(?:\d+(?:\.\d+)?|#-?\d+))\]", node.comment or "")
        cpl_match = re.search(r"(.+?) centipawn loss:\s*(\d+)\s*cp", node.comment or "")
        if eval_match:
            self.eval_label.config(text=f"Stockfish evaluation (White's perspective): {eval_match.group(1)}")
        elif node.comment:
            self.eval_label.config(text=f"Stockfish: {node.comment}")
        else:
            self.eval_label.config(text="Stockfish evaluation: not available in this PGN")
        if cpl_match:
            self.status_label.config(text=f"{cpl_match.group(1)} move centipawn loss: {cpl_match.group(2)} cp · Lower is better")
        elif self.position_index == 0:
            self.status_label.config(text="Use Next to step through the game or Auto-play to watch it.")
        else:
            self.status_label.config(text="Position evaluated by Stockfish during training.")

    def on_seek(self, value: str) -> None:
        if self._updating_seek or not self.positions:
            return
        target = max(0, min(len(self.positions) - 1, int(round(float(value)))))
        if target != self.position_index:
            self.stop_autoplay()
            self.position_index = target
            self.update_position()

    def previous_move(self) -> None:
        self.stop_autoplay()
        if self.positions and self.position_index > 0:
            self.position_index -= 1
            self.update_position()

    def next_move(self) -> None:
        if self.positions and self.position_index < len(self.positions) - 1:
            self.position_index += 1
            self.update_position()
        else:
            self.stop_autoplay()

    def go_start(self) -> None:
        self.stop_autoplay()
        if self.positions:
            self.position_index = 0
            self.update_position()

    def go_end(self) -> None:
        self.stop_autoplay()
        if self.positions:
            self.position_index = len(self.positions) - 1
            self.update_position()

    def toggle_autoplay(self) -> None:
        if self.autoplay_job is not None:
            self.stop_autoplay()
            return
        if not self.positions or self.position_index >= len(self.positions) - 1:
            self.go_start()
        self.autoplay_button.config(text="Ⅱ Pause")
        self._schedule_autoplay()

    def _schedule_autoplay(self) -> None:
        try:
            delay = int(float(self.speed_var.get().split()[0]) * 1000)
        except (ValueError, IndexError):
            delay = 1000
        self.autoplay_job = self.root.after(delay, self._autoplay_tick)

    def _autoplay_tick(self) -> None:
        self.autoplay_job = None
        if not self.positions or self.position_index >= len(self.positions) - 1:
            self.stop_autoplay()
            return
        self.position_index += 1
        self.update_position()
        self._schedule_autoplay()

    def stop_autoplay(self) -> None:
        if self.autoplay_job is not None:
            try:
                self.root.after_cancel(self.autoplay_job)
            except tk.TclError:
                pass
            self.autoplay_job = None
        if hasattr(self, "autoplay_button"):
            self.autoplay_button.config(text="▶ Auto-play")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = ReplayWindow()
    app.run()


if __name__ == "__main__":
    main()
