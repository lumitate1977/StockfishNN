from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from game_archive import ALL_GAMES_PATH, GOOD_GAMES_PATH
from export_pgn import export_games

ROOT = Path(__file__).resolve().parent
BG = "#20242b"
PANEL = "#2b313b"
FG = "#f4f4f5"
MUTED = "#aab6c5"


class MainMenu:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("StockfishNN — Control Center")
        self.root.geometry("760x690")
        self.root.minsize(680, 620)
        self.root.configure(bg=BG)
        self._build()

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=BG, padx=24, pady=22)
        header.pack(fill="x")
        tk.Label(header, text="StockfishNN", font=("Segoe UI", 25, "bold"), fg=FG, bg=BG).pack(anchor="w")
        tk.Label(header, text="Chess AI control center · play, train, analyze, and manage your games",
                 font=("Segoe UI", 10), fg=MUTED, bg=BG).pack(anchor="w", pady=(4, 0))

        body = tk.Frame(self.root, bg=BG, padx=22, pady=4)
        body.pack(fill="both", expand=True)
        self._section(body, "PLAY & REVIEW", [
            ("Play against your neural network", self.launch_play, "Open the chessboard and choose a checkpoint"),
            ("Replay archived games", self.launch_replay, "Review moves, evaluations, and centipawn loss"),
            ("Run a model-vs-model / Stockfish arena", self.launch_arena, "Play checkpoints against each other or Stockfish"),
        ])
        self._section(body, "TRAIN & EVALUATE", [
            ("Continue Stockfish-guided training", lambda: self.launch_script("train.py"), "Resume the main training run from its saved checkpoint"),
            ("Pretrain from human PGN games", lambda: self.launch_script("pretrain_human.py"), "Learn from master games in data/pgn"),
            ("Combine latest + human-pretrained models", self.merge_models, "Create models/combined.pt using a configurable weight blend"),
            ("Compare checkpoints against Stockfish", lambda: self.launch_script("evaluate_stockfish.py"), "Run the paired benchmark and save a CSV"),
        ])
        self._section(body, "GAME & FILE MANAGEMENT", [
            ("Export archived games to PGN", self.export_archive, "Export one combined PGN or one file per game"),
            ("Open games archive folder", lambda: self.open_folder(ROOT / "games"), "Find all_games.pgn and good_games.pgn"),
            ("Open model checkpoints folder", lambda: self.open_folder(ROOT / "models"), "Browse latest.pt and other model files"),
            ("Open project folder", lambda: self.open_folder(ROOT), "Open the StockfishNN project directory"),
        ])
        footer = tk.Frame(self.root, bg=BG, padx=22, pady=16)
        footer.pack(fill="x")
        ttk.Separator(footer).pack(fill="x", pady=(0, 12))
        tk.Label(footer, text="Shortcuts: Ctrl+P Play · Ctrl+R Replay · Ctrl+E Export PGN · Esc Close",
                 fg=MUTED, bg=BG, font=("Segoe UI", 9)).pack(anchor="w")
        self.root.bind("<Control-p>", lambda _e: self.launch_play())
        self.root.bind("<Control-r>", lambda _e: self.launch_replay())
        self.root.bind("<Control-e>", lambda _e: self.export_archive())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def _section(self, parent, title: str, actions: list[tuple[str, object, str]]) -> None:
        frame = tk.Frame(parent, bg=PANEL, padx=14, pady=10)
        frame.pack(fill="x", pady=6)
        tk.Label(frame, text=title, font=("Segoe UI", 9, "bold"), fg="#93c5fd", bg=PANEL).pack(anchor="w", pady=(0, 6))
        for label, command, description in actions:
            row = tk.Frame(frame, bg=PANEL)
            row.pack(fill="x", pady=3)
            button = ttk.Button(row, text=label, command=command, width=38)
            button.pack(side="left")
            tk.Label(row, text=description, fg=MUTED, bg=PANEL, font=("Segoe UI", 9), anchor="w").pack(side="left", padx=(12, 0), fill="x", expand=True)

    def _spawn(self, script: str, *args: str) -> None:
        path = ROOT / script
        if not path.is_file():
            messagebox.showerror("Missing feature", f"Could not find {path}", parent=self.root)
            return
        try:
            flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            subprocess.Popen([sys.executable, str(path), *args], cwd=str(ROOT), creationflags=flags)
        except Exception as exc:
            messagebox.showerror("Could not launch", str(exc), parent=self.root)

    def launch_script(self, script: str) -> None:
        self._spawn(script)

    def launch_play(self) -> None:
        self._spawn("play_gui.py")

    def launch_replay(self) -> None:
        self._spawn("replay_gui.py")

    def launch_arena(self) -> None:
        white = simpledialog.askstring("Arena", "White checkpoint (or 'stockfish'):", initialvalue="models/latest.pt", parent=self.root)
        if white is None:
            return
        black = simpledialog.askstring("Arena", "Black checkpoint (or 'stockfish'):", initialvalue="models/human_pretrained.pt", parent=self.root)
        if black is None:
            return
        games = simpledialog.askinteger("Arena", "Number of games:", initialvalue=10, minvalue=1, parent=self.root)
        if games is None:
            return
        self._spawn("arena.py", "--white", white.strip(), "--black", black.strip(), "--games", str(games))


    def merge_models(self) -> None:
        weight = simpledialog.askfloat(
            "Combine checkpoints",
            "How much should human_pretrained.pt contribute?\n\n0.0 = only latest.pt\n0.5 = equal blend\n1.0 = only human_pretrained.pt",
            initialvalue=0.5, minvalue=0.0, maxvalue=1.0, parent=self.root,
        )
        if weight is None:
            return
        output = filedialog.asksaveasfilename(
            parent=self.root, title="Save combined checkpoint",
            initialdir=str(ROOT / "models"), initialfile="combined.pt",
            defaultextension=".pt", filetypes=[("PyTorch checkpoints", "*.pt")],
        )
        if not output:
            return
        self._spawn(
            "merge_checkpoints.py",
            "--latest", "models/latest.pt",
            "--pretrained", "models/human_pretrained.pt",
            "--output", output,
            "--pretrained-weight", str(weight),
        )

    def export_archive(self) -> None:
        source = filedialog.askopenfilename(
            parent=self.root, title="Choose a PGN archive",
            initialdir=str(ROOT / "games"),
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")],
        )
        if not source:
            return
        split = messagebox.askyesno(
            "PGN export format",
            "Export each game as its own .pgn file?\n\nYes = separate files in a folder\nNo = one combined PGN file",
            parent=self.root,
        )
        if split:
            output = filedialog.askdirectory(parent=self.root, title="Choose export folder")
            if not output:
                return
            destination = Path(output)
        else:
            output = filedialog.asksaveasfilename(
                parent=self.root, title="Save combined PGN",
                defaultextension=".pgn", initialfile="stockfishnn_games.pgn",
                filetypes=[("PGN files", "*.pgn")],
            )
            if not output:
                return
            destination = Path(output)
        try:
            paths = export_games(Path(source), destination, split=split, overwrite=False)
        except Exception as exc:
            messagebox.showerror("PGN export failed", str(exc), parent=self.root)
            return
        messagebox.showinfo(
            "PGN export complete",
            f"Exported {len(paths)} file(s).\n\nLocation:\n{destination}",
            parent=self.root,
        )

    def open_folder(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                import os
                os.startfile(str(path))
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Could not open folder", str(exc), parent=self.root)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    MainMenu().run()


if __name__ == "__main__":
    main()
