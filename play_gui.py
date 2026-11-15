from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from pathlib import Path

import chess
import torch

from chess_ai import ROOT, choose_move, load_model


class ChessWindow:
    SQUARE = 76
    BOARD_SIZE = SQUARE * 8
    LIGHT = "#f0d9b5"
    DARK = "#b58863"
    SELECTED = "#f6f669"
    LAST = "#cdd26a"
    DEST = "#4f7f52"
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

    def __init__(self, model, device, metadata) -> None:
        self.model = model
        self.device = device
        self.metadata = metadata
        self.board = chess.Board()
        self.human_color = chess.WHITE
        self.selected: chess.Square | None = None
        self.last_move: chess.Move | None = None
        self.ai_busy = False

        self.root = tk.Tk()
        self.root.title("StockfishNN — Play Against Your Neural Network")
        self.root.resizable(False, False)
        self.root.configure(bg="#20242b")

        header = tk.Frame(self.root, bg="#20242b", padx=12, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="StockfishNN",
            font=("Segoe UI", 18, "bold"),
            fg="#f4f4f5",
            bg="#20242b",
        ).pack(side="left")
        tk.Label(header, text="You play vs Neural Network (not Stockfish)", fg="#cbd5e1", bg="#20242b").pack(side="left", padx=(18, 6))
        self.color_var = tk.StringVar(value="White")
        color_menu = ttk.Combobox(header, textvariable=self.color_var, values=["White", "Black"], width=8, state="readonly")
        color_menu.pack(side="left")
        color_menu.bind("<<ComboboxSelected>>", self.change_color)

        self.canvas = tk.Canvas(
            self.root,
            width=self.BOARD_SIZE,
            height=self.BOARD_SIZE,
            highlightthickness=0,
            bg="#20242b",
        )
        self.canvas.pack(padx=12, pady=(0, 8))
        self.canvas.bind("<Button-1>", self.on_click)

        controls = tk.Frame(self.root, bg="#20242b", padx=12, pady=8)
        controls.pack(fill="x")
        ttk.Button(controls, text="New game", command=self.new_game).pack(side="left")
        ttk.Button(controls, text="Undo turn", command=self.undo_turn).pack(side="left", padx=8)
        ttk.Button(controls, text="Replay archive", command=self.open_replay).pack(side="left", padx=8)
        ttk.Button(controls, text="Load checkpoint…", command=self.load_checkpoint).pack(side="left", padx=8)
        self.status = tk.Label(
            self.root,
            text="Your move",
            font=("Segoe UI", 11),
            fg="#f4f4f5",
            bg="#20242b",
            anchor="w",
            padx=14,
            pady=10,
        )
        self.status.pack(fill="x")
        self.info = tk.Label(
            self.root,
            text=f"Model checkpoint: {metadata.get('cycles_completed', 0)} training cycles",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#20242b",
            anchor="w",
            padx=14,
            pady=10,
        )
        self.info.pack(fill="x")

        self.draw_board()
        self.root.after(200, self.maybe_start_ai)

    def orient(self, square: chess.Square) -> tuple[int, int]:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if self.human_color == chess.WHITE:
            return file_index, 7 - rank_index
        return 7 - file_index, rank_index

    def square_at(self, x: int, y: int) -> chess.Square | None:
        file_index = x // self.SQUARE
        row = y // self.SQUARE
        if not (0 <= file_index < 8 and 0 <= row < 8):
            return None
        if self.human_color == chess.WHITE:
            return chess.square(file_index, 7 - row)
        return chess.square(7 - file_index, row)

    def draw_board(self) -> None:
        self.canvas.delete("all")
        legal_targets = set()
        if self.selected is not None:
            legal_targets = {move.to_square for move in self.board.legal_moves if move.from_square == self.selected}

        for square in chess.SQUARES:
            file_index, row = self.orient(square)
            x0, y0 = file_index * self.SQUARE, row * self.SQUARE
            x1, y1 = x0 + self.SQUARE, y0 + self.SQUARE
            color = self.LIGHT if (chess.square_file(square) + chess.square_rank(square)) % 2 else self.DARK
            if self.last_move and square in (self.last_move.from_square, self.last_move.to_square):
                color = self.LAST
            if square == self.selected:
                color = self.SELECTED
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

            piece = self.board.piece_at(square)
            if piece:
                glyph = self.PIECES[piece.color][piece.piece_type]
                fill = "#fffdf4" if piece.color == chess.WHITE else "#171717"
                outline = "#282828" if piece.color == chess.WHITE else "#eee8dc"
                # Segoe UI Symbol's black pawn has a different visual baseline
                # from the other chess glyphs. Draw it slightly smaller and higher
                # so it sits optically centered in its square.
                is_black_pawn = piece.color == chess.BLACK and piece.piece_type == chess.PAWN
                self.canvas.create_text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2 - (3 if is_black_pawn else 0),
                    text=glyph,
                    font=("Segoe UI Symbol", 43 if is_black_pawn else 48),
                    fill=fill,
                    activefill=fill,
                )
            elif square in legal_targets:
                self.canvas.create_oval(
                    x0 + 28, y0 + 28, x1 - 28, y1 - 28,
                    fill=self.DEST, outline=self.DEST,
                )

        for file_index in range(8):
            label_file = file_index if self.human_color == chess.WHITE else 7 - file_index
            label_rank = 7 if self.human_color == chess.WHITE else 0
            x = file_index * self.SQUARE + self.SQUARE - 9
            y = 7 * self.SQUARE + self.SQUARE - 12
            self.canvas.create_text(x, y, text=chess.FILE_NAMES[label_file], fill="#333333", font=("Segoe UI", 9, "bold"))
        for row in range(8):
            label_rank = 7 - row if self.human_color == chess.WHITE else row
            self.canvas.create_text(8, row * self.SQUARE + 9, text=str(label_rank + 1), fill="#333333", font=("Segoe UI", 9, "bold"))

    def on_click(self, event) -> None:
        if self.ai_busy or self.board.is_game_over(claim_draw=True) or self.board.turn != self.human_color:
            return
        square = self.square_at(event.x, event.y)
        if square is None:
            return
        piece = self.board.piece_at(square)
        if self.selected is None:
            if piece and piece.color == self.human_color:
                self.selected = square
                self.draw_board()
            return

        if piece and piece.color == self.human_color:
            self.selected = square
            self.draw_board()
            return

        choices = [move for move in self.board.legal_moves if move.from_square == self.selected and move.to_square == square]
        if not choices:
            self.selected = None
            self.draw_board()
            return
        move = choices[0]
        if len(choices) > 1 and any(item.promotion for item in choices):
            answer = simpledialog.askstring("Promote pawn", "Promote to Q, R, B, or N:", initialvalue="Q", parent=self.root)
            if answer is None:
                return
            promotion_map = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}
            move = next((item for item in choices if item.promotion == promotion_map.get(answer.strip().upper())), choices[-1])
        self.board.push(move)
        self.last_move = move
        self.selected = None
        self.update_status()
        self.draw_board()
        self.maybe_start_ai()

    def maybe_start_ai(self) -> None:
        if self.board.is_game_over(claim_draw=True):
            self.update_status()
            return
        if self.board.turn != self.human_color and not self.ai_busy:
            self.ai_busy = True
            self.status.config(text="Your neural network is thinking…")
            self.root.update_idletasks()
            try:
                move = choose_move(self.model, self.board, self.device, temperature=0.0)
                self.board.push(move)
                self.last_move = move
            except Exception as exc:
                messagebox.showerror("AI move error", str(exc), parent=self.root)
            finally:
                self.ai_busy = False
            self.update_status()
            self.draw_board()

    def update_status(self) -> None:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                self.status.config(text=f"Game over — draw ({outcome.result()})")
            else:
                winner = "White" if outcome.winner == chess.WHITE else "Black"
                self.status.config(text=f"Game over — {winner} wins ({outcome.result()})")
        elif self.board.turn == self.human_color:
            self.status.config(text="Your move — select a piece, then a destination")
        else:
            self.status.config(text="Neural network to move")

    def new_game(self) -> None:
        self.board = chess.Board()
        self.selected = None
        self.last_move = None
        self.update_status()
        self.draw_board()
        self.maybe_start_ai()

    def change_color(self, _event=None) -> None:
        self.human_color = chess.WHITE if self.color_var.get() == "White" else chess.BLACK
        self.new_game()

    def undo_turn(self) -> None:
        if not self.board.move_stack:
            return
        # With Black, the engine's opening move is not a completed human turn.
        if self.human_color == chess.BLACK and len(self.board.move_stack) == 1:
            return
        # Undo the latest AI reply and the human move that preceded it.
        self.board.pop()
        if self.board.move_stack:
            self.board.pop()
        self.selected = None
        self.last_move = self.board.peek() if self.board.move_stack else None
        self.update_status()
        self.draw_board()
        self.maybe_start_ai()


    def load_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a StockfishNN checkpoint",
            initialdir=str(ROOT / "models"),
            filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            model, device, metadata = load_model(path)
        except Exception as exc:
            messagebox.showerror("Could not load checkpoint", str(exc), parent=self.root)
            return
        self.model, self.device, self.metadata = model, device, metadata
        self.new_game()
        self.info.config(
            text=f"Checkpoint: {Path(path).name} | "
                 f"{metadata.get('cycles_completed', metadata.get('games_trained', 0))} training cycles/games"
        )

    def open_replay(self) -> None:
        replay_script = ROOT / "replay_gui.py"
        if not replay_script.is_file():
            messagebox.showerror("Replay viewer missing", f"Could not find {replay_script}", parent=self.root)
            return
        try:
            subprocess.Popen([sys.executable, str(replay_script)], cwd=str(ROOT))
        except Exception as exc:
            messagebox.showerror("Could not open replay viewer", str(exc), parent=self.root)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        model, device, metadata = load_model()
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Could not load model", str(exc))
        root.destroy()
        return
    app = ChessWindow(model, device, metadata)
    app.run()


if __name__ == "__main__":
    main()
