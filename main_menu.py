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
            ("Pretrain checkmates from puzzles", lambda: self.launch_script("pretrain_checkmates.py"), "Learn mate-in-1, mate-in-2, and mate-in-3 patterns"),
            ("Combine selected models (weighted)", self.merge_models, "Choose any number of checkpoints and set their blend percentages"),
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
        models_dir = ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        available = sorted(models_dir.glob("*.pt"))
        if len(available) < 2:
            messagebox.showerror("Combine models", "At least two .pt checkpoints must exist in the models folder.", parent=self.root)
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Combine neural-network checkpoints")
        dialog.configure(bg=BG)
        dialog.geometry("680x650")
        dialog.minsize(580, 480)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Choose checkpoints and blend percentages",
                 bg=BG, fg=FG, font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(dialog, text="Select at least two models. Their percentages must total exactly 100%.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=18, pady=(0, 10))

        canvas_frame = tk.Frame(dialog, bg=BG)
        canvas_frame.pack(fill="both", expand=True, padx=14, pady=4)
        canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        entries = []
        default_paths = ["latest.pt", "human_pretrained.pt", "mate_pretrained.pt"]
        existing_defaults = [p for p in available if p.name in default_paths]
        other_paths = [p for p in available if p not in existing_defaults]
        ordered = existing_defaults + other_paths
        for index, path in enumerate(ordered):
            default_selected = path.name in default_paths and path.name in {
                "latest.pt", "human_pretrained.pt", "mate_pretrained.pt"
            }
            selected = tk.BooleanVar(value=default_selected)
            weight = tk.IntVar(value=34 if index == 0 else 33 if index < 3 else 0)
            row = tk.Frame(inner, bg=PANEL, padx=10, pady=8)
            row.pack(fill="x", pady=3)
            check = tk.Checkbutton(row, text=path.name, variable=selected, bg=PANEL, fg=FG,
                                   selectcolor=BG, activebackground=PANEL, activeforeground=FG,
                                   font=("Segoe UI", 10, "bold"))
            check.pack(anchor="w")
            value_label = tk.Label(row, text=f"{weight.get()}%", bg=PANEL, fg="#93c5fd", width=6)
            value_label.pack(side="right")
            scale = tk.Scale(row, from_=0, to=100, orient="horizontal", variable=weight,
                             showvalue=False, resolution=1, length=440, bg=PANEL, fg=FG,
                             troughcolor="#46505e", highlightthickness=0,
                             command=lambda value, label=value_label: label.config(text=f"{int(float(value))}%"))
            scale.pack(fill="x")
            entries.append((path, selected, weight))

        total_var = tk.StringVar()
        status_var = tk.StringVar()
        def update_total(*_args):
            total = sum(weight.get() for _path, selected, weight in entries if selected.get())
            count = sum(1 for _path, selected, _weight in entries if selected.get())
            total_var.set(f"Selected: {count} models     Total: {total}%")
            status_var.set("Percentages must total 100%." if total != 100 else "Ready to combine.")
        for _path, selected, weight in entries:
            selected.trace_add("write", update_total)
            weight.trace_add("write", update_total)
        update_total()

        footer = tk.Frame(dialog, bg=BG, padx=18, pady=12)
        footer.pack(fill="x")
        tk.Label(footer, textvariable=total_var, bg=BG, fg=FG,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(footer, textvariable=status_var, bg=BG, fg=MUTED).pack(anchor="w", pady=(2, 8))

        def combine():
            chosen = [(path, weight.get()) for path, selected, weight in entries if selected.get()]
            if len(chosen) < 2:
                messagebox.showerror("Combine models", "Select at least two checkpoints.", parent=dialog)
                return
            if sum(weight for _path, weight in chosen) != 100:
                messagebox.showerror("Combine models", "Selected percentages must add up to exactly 100%.", parent=dialog)
                return
            if sum(weight > 0 for _path, weight in chosen) < 2:
                messagebox.showerror("Combine models", "At least two selected models need a non-zero percentage.", parent=dialog)
                return
            output = filedialog.asksaveasfilename(
                parent=dialog, title="Save combined checkpoint",
                initialdir=str(models_dir), initialfile="combined.pt",
                defaultextension=".pt", filetypes=[("PyTorch checkpoints", "*.pt")],
            )
            if not output:
                return
            args = []
            for path, weight in chosen:
                args.extend(["--model", f"{path}={weight}"])
            args.extend(["--output", output])
            dialog.destroy()
            self._spawn("merge_checkpoints.py", *args)

        buttons = tk.Frame(footer, bg=BG)
        buttons.pack(anchor="e")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(buttons, text="Combine models", command=combine).pack(side="right")

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
