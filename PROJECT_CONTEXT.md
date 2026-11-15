# StockfishNN — Project Context

Last updated: 2026-11-14

## 1. Project overview

**Project:** StockfishNN  
**Local project directory:** `C:\Users\thoma\Downloads\StockfishNN`  
**GitHub repository:** https://github.com/lumitate1977/StockfishNN  
**Default branch:** `main`

The goal is to build a chess-playing neural network in PyTorch. It learns from Stockfish supervision, human/master PGNs, move-level and game-level rewards, replay training, and eventually self-play. The long-term goal is for the network to become a capable independent chess player.

## 2. Hardware and environment

- OS: Windows
- CPU: AMD Ryzen 7 5800X
- GPU: AMD Radeon RX 6600 8GB
- RAM: 32GB DDR4
- Python: 3.13
- Framework: PyTorch
- Chess library: python-chess
- Training: currently primarily CPU-based
- Teacher/opponent: external Stockfish UCI engine

The RX 6600 is not supported by standard CUDA PyTorch builds. Do not assume CUDA is available; check the user's actual PyTorch backend before changing device configuration.

## 3. Important files

- `chess_ai.py` — ChessNet architecture, board encoding, move encoding, move selection, and model/config loading.
- `train.py` — Main Stockfish-guided training loop, replay training, rewards, evaluation, and checkpoint saving.
- `game_archive.py` — Archives training games and game metrics.
- `pretrain_human.py` — Separate supervised pretraining from human/master PGN files.
- `evaluate_stockfish.py` — Paired benchmark comparing two checkpoints against the same Stockfish settings.
- `play_gui.py` — GUI for playing against the network.
- `replay.py` / `replay_gui.py` — Replay and game-review functionality.
- `config.json` — Paths and project configuration.
- `models/latest.pt` — Main/latest checkpoint.
- `models/cycle_XXXX.pt` — Historical cycle checkpoints.
- `models/human_pretrained.pt` — Default output of the separate human-PGN pretraining script.
- `data/replay_buffer_move_game_v3.npz` — Existing replay buffer.
- `training_log_eval_v2.csv` — Training/evaluation statistics.
- `PROJECT_CONTEXT.md` — This project state and handoff document.

Other launcher scripts and data files may also exist.

## 4. Neural network architecture

The current `ChessNet` architecture in `chess_ai.py` uses:

- Input: 18 board planes.
- Convolutional stem: 48 channels.
- Two residual blocks.
- Shared fully connected layer: 256 units.
- Policy head: 20,480 possible move actions.
- Value head: scalar position evaluation in the range [-1, 1].

### Board encoding

The 18 planes represent:
- 12 piece planes: six piece types for each color.
- Four castling-right planes.
- One side-to-move plane.
- One en-passant plane.

### Move encoding

`ACTION_SIZE = 5 * 64 * 64 = 20480`

The five promotion slots are:
1. Normal move.
2. Knight promotion.
3. Bishop promotion.
4. Rook promotion.
5. Queen promotion.

Move indices use the promotion slot, source square, and destination square. Illegal moves are masked during move selection/training where supported.

**Compatibility rule:** Do not change the board encoding, move encoding, action size, or network architecture casually. Any architecture change needs a checkpoint migration plan and must preserve the existing checkpoint files.

## 5. Existing Stockfish-guided training

The main training system includes:
- Stockfish as teacher and opponent.
- Neural-network move selection.
- Stockfish position evaluations.
- Individual move-quality rewards.
- Whole-game result rewards.
- Replay-buffer training.
- Policy and value losses, plus a reinforcement-style loss.
- Periodic checkpoints and training statistics.
- Optional benchmark games.

The following are documented command-line defaults from the current project context. Verify the current `train.py --help` or source before relying on them, because arguments/defaults may have changed:

| Setting | Previously documented default |
|---|---:|
| Cycles | 10 |
| Games per cycle | 8 |
| Stockfish depth | 5 |
| Stockfish skill | 5 |
| Epochs per cycle | 2 |
| Batch size | 128 |
| Maximum plies | 120 |
| Replay capacity | 30,000 |
| Samples per epoch | 2,500 |
| Stockfish hash | 512 MB |
| PyTorch threads | 4 |
| Learning rate | 0.0003 |
| Reward scale | 0.25 |

These are defaults, not a record of the settings used in the user's last training run.

## 6. Reward system

The current implementation previously inspected in `train.py` combines:
- 60% individual move-quality reward.
- 40% whole-game reward.

The console message was noted to incorrectly display 75%/25%; the actual blend in `train_epoch()` was 60%/40%. Check the current source before making further reward changes.

Move quality is estimated from Stockfish evaluation changes and an approximate centipawn-loss calculation. This is not necessarily a full best-move-versus-played-move CPL calculation. Material-loss penalties may also be applied.

Completed-game outcomes are:
- Win: +1
- Draw: 0
- Loss: -1

Unfinished games use a final Stockfish evaluation rather than automatically being treated as draws.

The previously documented combined loss was:

```python
total_loss = supervised_loss + 0.5 * value_loss + 0.05 * reinforce_loss
```

Verify the live implementation before editing this formula.

## 7. Replay buffer and checkpoints — preserve these

### Existing replay data

`data/replay_buffer_move_game_v3.npz`

The buffer stores accumulated training experience, including encoded positions, teacher actions, value targets, legal actions, NN-selected actions, move rewards, and game rewards (depending on the stored buffer version).

### Existing model files

- Main checkpoint: `models/latest.pt`
- Historical checkpoints: `models/cycle_XXXX.pt`
- Human-pretraining output: `models/human_pretrained.pt`

The checkpoint format previously documented contains:
- `model_state`
- `optimizer_state`
- `metadata`

The main training program resumes from `models/latest.pt` by default unless `--fresh` is used.

**Critical data-safety rules:**
1. Never delete, reset, or overwrite `models/latest.pt`, historical checkpoints, or the replay buffer unless the user explicitly requests it.
2. Never recommend `--fresh` as a routine fix. It is intended to start fresh and may reset training state.
3. Back up important files before risky migrations or bulk edits.
4. Keep human pretraining output separate until it has been evaluated.
5. Do not claim a local checkpoint or replay file was changed; GitHub access does not modify the user's local machine.

## 8. Human/master PGN pretraining — implemented in the repository

The repository contains `pretrain_human.py`. It is a separate supervised-learning stage intended to teach the network from recorded human/master games before returning to Stockfish-guided training.

### Input data

Place PGN files in:

```
data/
  pgn/
    Carlsen.pgn
    Caruana.pgn
    Anand.pgn
    Fischer.pgn
```

The script supports standard `.pgn` and gzip-compressed `.pgn.gz` files and searches the input directory recursively. If a download is a ZIP archive, extract it first; ZIP input is not documented as supported.

The proposed initial collection from PGN Mentor includes games by Carlsen, Caruana, Anand, and Fischer, with selected World Championship collections as optional additions. The user reports that PGN training has completed locally. The expected candidate checkpoint is `models/human_pretrained.pt`; exact local file presence and training log have not been independently verified.

PGN source previously recommended:
https://www.pgnmentor.com/files.html

### What the script does

The script is designed to:
1. Read PGN games and use each game's mainline.
2. Skip games with unfinished results.
3. Encode each position using the existing 18-plane board encoder.
4. Use the recorded human move as the policy target.
5. Build legal-move masks so illegal actions do not contribute to policy training.
6. Use the final game result as a value target from the side-to-move perspective.
7. Fine-tune the existing ChessNet architecture.

This is imitation learning from recorded games. It complements, but does not replace, Stockfish's position-level evaluations, replay training, or self-play.

### Safe checkpoint behavior

- Default base checkpoint: configured model path, previously `models/latest.pt`.
- Default output: `models/human_pretrained.pt`.
- The script is designed to reject using the same path for input and output.
- It refuses to overwrite an existing output unless `--overwrite` is supplied.
- It does not intentionally modify the main checkpoint or replay buffer.

The script was committed to GitHub. The user reports PGN pretraining completed locally. Treat completion as user-reported; the assistant has not independently inspected the local checkpoint or training log.

### Intended first-run commands

From the project directory in PowerShell:

```powershell
New-Item -ItemType Directory -Force .\data\pgn
```

After placing PGN files in that folder:

```powershell
python .\pretrain_human.py --pgn-dir .\data\pgn --epochs 3 --max-positions 100000
```

Optional rating filter (requires suitable Elo tags in the PGNs):

```powershell
python .\pretrain_human.py --pgn-dir .\data\pgn --min-elo 2200 --epochs 3
```

If `models/human_pretrained.pt` already exists, use a different `--output` path or pass `--overwrite` only if replacing that separate pretraining output is intentional.

### PGN pretraining status

- [x] Add separate PGN pretraining script to the GitHub repository.
- [x] Reuse existing board and move encodings.
- [x] Include legal-action masking in the supervised policy objective.
- [x] Support PGN and .pgn.gz input.
- [x] Save to a separate checkpoint by default.
- [x] User reports downloading PGNs and completing the local PGN pretraining run.
- [x] User reports that the PGN training run completed; local logs/checkpoint have not been independently verified.
- [x] Add `evaluate_stockfish.py` for a simple checkpoint-vs-Stockfish comparison.
- [x] Add `arena.py` for neural-network-vs-neural-network and neural-network-vs-Stockfish games.
- [x] Add Stockfish same-root move analysis and per-move CPL annotations to arena PGNs.
- [x] Make arena PGNs compatible with the existing replay archive.
- [x] Update replay list/details to display both player names and per-side CPL.
- [x] Add checkpoint selection to the Play GUI so the user can play against any `.pt` checkpoint.
- [x] Add optional arena policy-gradient training that writes to separate output checkpoints and protects source checkpoints.
- [ ] Run the new arena, GUI, and training features locally; they have not been locally tested.
- [ ] Decide whether to use the pretrained checkpoint as the base for further Stockfish/RL training.

### Comparing the original and PGN-pretrained checkpoints

The repository now includes `evaluate_stockfish.py`. It loads two checkpoints independently and plays each against the same Stockfish engine configuration. The default comparison is:
- Baseline: `models/latest.pt`
- Candidate: `models/human_pretrained.pt`

It alternates the NN's color across games, uses greedy NN move selection (temperature 0), applies the same Stockfish depth/skill/threads/hash to both checkpoints, and saves per-game results to `evaluation_results.csv`. It does not train or save either model. Unfinished games are reported separately and excluded from the completed-game score percentage.

Default command from the project directory:

```powershell
python .\\evaluate_stockfish.py --games-per-model 10
```

For a larger sample after the initial check:

```powershell
python .\\evaluate_stockfish.py --games-per-model 30
```

The defaults read Stockfish skill, depth, threads, hash, and executable path from `config.json`. You can override these with `--depth`, `--skill`, `--threads`, and `--hash-mb`. Both checkpoints must exist locally. The baseline is expected to be the untouched pre-PGN `models/latest.pt`; do not run normal training or promote the candidate before completing this comparison.

A small 10-game-per-model result is an initial signal, not a statistically strong rating estimate. If results are close, increase to 30–50 games per model and consider a second test at a stronger fixed Stockfish setting.

### Arena matches, detailed analysis, and arena training

The repository now contains `arena.py`. It supports:
- Neural network vs neural network.
- Neural network vs Stockfish.
- Detailed PGN archiving into `games/all_games.pgn` (or a custom `--archive`).
- Stockfish evaluation annotations and per-move CPL.
- Optional training of the participating neural network(s).

CPL is calculated by comparing Stockfish's unrestricted best evaluation at a position with a second search at the same root restricted to the move actually played. This is more meaningful than simply comparing evaluations before and after a move. It remains dependent on analysis depth and can be noisy at shallow depth.

Examples from the project directory:

```powershell
# Compare the original and PGN-pretrained networks; archive games for replay
python .\\arena.py --white models/latest.pt --black models/human_pretrained.pt --games 10

# PGN-pretrained network against Stockfish
python .\\arena.py --white models/human_pretrained.pt --black stockfish --games 10

# Train both networks against each other; source checkpoints are preserved
python .\\arena.py --white models/latest.pt --black models/human_pretrained.pt --games 100 --train --white-output models/arena_latest.pt --black-output models/arena_human.pt

# Train the PGN model against Stockfish into a separate output checkpoint
python .\\arena.py --white models/human_pretrained.pt --black stockfish --games 100 --train --white-output models/human_vs_stockfish.pt
```

Arena games alternate the neural-network players' colors across successive games to reduce first-player/color bias. Arena training applies a simple policy-gradient update to each participating model after each game. The reward blends game outcome with the move's Stockfish CPL quality. This is an experimental baseline, not a mature self-play algorithm or a guarantee of improvement. Start with a small run (e.g. 10–20 games) and compare the output checkpoint against its source before scaling up. Existing input checkpoints are never overwritten by the default arena outputs; if an output file already exists, the script refuses to overwrite it unless `--overwrite-output` is explicitly supplied.

The replay viewer already reads PGN archives. Arena games carry player names, result, average/max CPL for both colors, and per-move Stockfish evaluation/CPL comments. The replay UI has been updated to display those fields.

The Play GUI now has a **Load checkpoint…** button. Start `play_gui.py`, choose any checkpoint under `models` (including `human_pretrained.pt`, `arena_white.pt`, etc.), and start a new game against that model.

None of these new features has yet been run on the user's local PC. Confirm local Git synchronization, install dependencies if needed, and smoke-test with one or two games before a longer run. Arena games are appended to the selected PGN archive; they do not change model checkpoints unless `--train` is used.

## 9. Chess knowledge curriculum

The planned learning sequence is:

1. **Master games:** supervised learning from high-quality human games.
2. **Tactics:** puzzle positions and tactical motifs, with accurate solution targets.
3. **Openings:** opening principles and curated opening positions; avoid simply memorizing moves without position understanding.
4. **Endgames:** basic endgame knowledge, then tablebase-supervised positions where practical.
5. **Positional play:** pawn structure, piece activity, king safety, weak squares, and plans.
6. **Stockfish guidance:** position evaluation and move-quality feedback.
7. **Self-play:** practice against older frozen network versions and progressively stronger opponents.

Do not attempt to add all curriculum stages at once. Start with PGN pretraining, test it, and measure whether it improves play before adding more data sources or changing the model.

## 10. Planned training improvements

These remain plans unless verified in the current source:

### A. Mixed Stockfish difficulty

Use a pool of Stockfish strengths (previously proposed: skill 5, 8, and 12), selected per training game. Keep benchmark strength fixed so results remain comparable.

### B. Self-play against older network checkpoints

Use selected historical cycle checkpoints as frozen opponents. The older models must not receive gradients or be overwritten. Stockfish can continue evaluating positions and supplying teacher information.

A 20% self-play probability was previously proposed as an initial experiment, not confirmed as implemented.

### C. Better benchmarking

Use a fixed opponent strength, alternate NN colors, and record:
- Wins, draws, and losses.
- Unfinished games.
- Win rate.
- Average centipawn loss, where measured consistently.
- Performance over time.

Keep benchmark results separate from ordinary training games.

## 11. Git and synchronization status

The GitHub repository is `lumitate1977/StockfishNN`.

The PGN pretraining script and its documentation were committed to GitHub in commit:

`e19a05c254955ae13e91ce5a503d2b0d9846f394`

The user previously ran `git pull`, which fetched the remote update but aborted because local uncommitted changes to `PROJECT_CONTEXT.md` would have been overwritten. The suggested next command was:

```powershell
git stash push -m "Save local context changes"
git pull
```

The user has not yet confirmed whether the stash and pull completed. Do not assume the local working copy is synchronized.

If the stash was created, do not blindly run `git stash pop`: the remote and local context files may conflict. Inspect the stash and reconcile changes deliberately. Never use `git reset --hard` or discard the stash as a shortcut.

## 12. Immediate next steps

1. Synchronize the local working copy with GitHub safely; check the earlier stash/pull state before popping or discarding anything.
2. Confirm `models/latest.pt` and `models/human_pretrained.pt` exist locally.
3. Run a short model-vs-model arena match:
   ```powershell
   python .\\arena.py --white models/latest.pt --black models/human_pretrained.pt --games 2
   ```
4. Open `replay.bat` and verify the new match appears with per-move CPL and evaluations.
5. Open `play.bat`, use **Load checkpoint…**, and select the human-pretrained checkpoint.
6. After the arena path is verified, try a small training run to separate output checkpoints; compare the trained copies against their source checkpoints before increasing games.
7. Only after these checks, decide how to integrate arena training with the main Stockfish replay training and expand the tactics/endgame curriculum.

## 13. User preferences and implementation rules

- Provide complete, usable code and exact Windows PowerShell commands.
- Prefer making and committing repository changes directly when GitHub access is available.
- Explain what changed and what the user must do locally.
- Avoid unnecessary back-and-forth; ask only for missing information needed to proceed.
- Preserve all existing model checkpoints and replay data by default.
- Do not claim code has been tested locally unless the user has actually run it or a suitable test environment was used.
- Do not claim GitHub edits have been pulled onto the user's PC.
- Whenever the project changes, update this `PROJECT_CONTEXT.md` so it accurately records the current state, what is implemented, what is untested, and the next steps.


## 14. Unified control center and PGN export (2026-11-14)

A central Windows launcher and archive-export utility have been added to the GitHub repository.

### New files
- `main_menu.py` — Tkinter control center for the project.
- `main.bat` — Windows launcher. It prefers `.venv\\Scripts\\python.exe` when a project virtual environment exists, otherwise uses `python`.
- `export_pgn.py` — Export archived games into standard PGN files.

### Main menu features
The menu groups project functions into:
- **Play & Review:** open the Play GUI, open the replay viewer, or configure an arena match.
- **Train & Evaluate:** launch the main Stockfish-guided training script, human-PGN pretraining, and checkpoint-vs-Stockfish evaluation.
- **Game & File Management:** export archived games, open the games folder, open the models folder, and open the project folder.

Arena launch asks for White and Black checkpoint paths (or `stockfish`) and the number of games. The play GUI retains its own checkpoint chooser. The menu launches scripts in separate console windows where applicable, so training/arena output remains visible.

Keyboard shortcuts in the menu:
- Ctrl+P: Play
- Ctrl+R: Replay
- Ctrl+E: Export PGN
- Esc: Close the menu

### PGN extraction/export
The menu's **Export archived games to PGN** action:
1. Lets the user choose an existing archive, normally `games/all_games.pgn` or `games/good_games.pgn`.
2. Offers either one combined PGN file or separate PGN files for every archived game.
3. Asks for the output filename or destination folder.
4. Refuses to overwrite existing files by default.

Command-line examples from the project root:

```powershell
# Export the full archive into one standalone PGN
python .\\export_pgn.py --source games\\all_games.pgn --output exports\\all_games.pgn

# Export one PGN file per game
python .\\export_pgn.py --source games\\all_games.pgn --output exports\\individual --split

# Export only the curated good-games archive
python .\\export_pgn.py --good-games --output exports\\good_games.pgn
```

The exporter preserves PGN headers, move notation, variations, and comments, including Stockfish evaluation/CPL annotations stored in the games. It reads archives and writes new output files; it does not delete or modify the original archives, checkpoints, or replay buffer.

### Local setup / launch
After syncing the GitHub changes, run from PowerShell in the project directory:

```powershell
.\main.bat
```

If dependencies have not yet been installed in the active Python environment, install them with:

```powershell
python -m pip install -r requirements.txt
```

### Verification status
These new files have been committed to GitHub but have not been executed on the user's Windows PC. The user must pull the changes locally before `main.bat` will exist in their local project. First smoke test: open the menu, launch Replay, then export a small archive to a new folder and verify the PGN opens in a chess PGN viewer.


## 15. Combining latest and human-pretrained checkpoints (2026-11-14)

Added `merge_checkpoints.py` and a matching action in `main_menu.py`.

### What it does
- Loads `models/latest.pt` and `models/human_pretrained.pt`.
- Verifies both contain matching model-state keys, tensor shapes, and dtypes.
- Linearly interpolates floating-point model parameters and buffers:
  `combined = latest * (1 - human_weight) + human_pretrained * human_weight`.
- Keeps non-floating buffers (such as BatchNorm counters) from `latest.pt`.
- Saves to a separate output, default `models/combined.pt`.
- Never overwrites either source checkpoint.
- Does not combine optimizer states. If training resumes from the merged model, train.py creates a fresh optimizer; existing replay data remains as-is.

### Usage
From the project root:

```powershell
# Equal 50/50 blend
python .\\merge_checkpoints.py

# 70% latest + 30% human-pretrained
python .\\merge_checkpoints.py --pretrained-weight 0.3 --output models\\combined_70_latest.pt
```

The `--pretrained-weight` value is the contribution from `human_pretrained.pt`: 0.0 means only latest, 0.5 means equal blend, and 1.0 means only human-pretrained.

The main menu now includes **Combine latest + human-pretrained models** under Train & Evaluate. It asks for the human-pretrained contribution and output filename, then runs the merge in a separate console.

### Important limitations
This is parameter interpolation, not a true knowledge-preserving merge and not guaranteed to improve strength. Although both checkpoints use the same ChessNet architecture and likely share a common initialization, their training paths may have diverged. Benchmark `models/combined.pt` against both source checkpoints before using it as the primary model. Keep `models/latest.pt`, `models/human_pretrained.pt`, and replay data unchanged until the merged model proves better.

The new script and menu integration are committed to GitHub but have not been executed on the user's PC.

## 16. Replay board black-pawn alignment fix (2026-11-14)

Updated `replay.py` to remove the black-pawn-specific font-size reduction and vertical offset. Every chess piece is now drawn with the same Segoe UI Symbol font size (42) and the same exact square-center anchor, with explicit center anchoring/justification. The prior special-case adjustment could make black pawn glyphs appear displaced on some Windows font builds.

- This changes only board rendering; PGN data, move positions, evaluations, and archives are untouched.
- Committed to GitHub in commit `5f96c18ca6c421e1d0d5f840eca6cc35d1b8dd61`.
- Not yet verified on the user's Windows PC. Pull the update and reopen Replay to check the pawn alignment.


## 17. Checkmate-puzzle pretraining (2026-11-14)

Added \`pretrain_checkmates.py\` for a separate supervised tactical curriculum using the Lichess puzzle database CSV format.

### What it teaches
- Mate-in-one, mate-in-two, and mate-in-three patterns by default (Lichess themes \`mateIn1\`, \`mateIn2\`, \`mateIn3\`).
- Learner-to-move positions from the puzzle solution line.
- The policy head is trained to select the solution move, with illegal actions masked from cross-entropy.
- Only complete solution lines ending in actual checkmate are accepted.
- The puzzle's initial opponent setup move is skipped by default, matching the standard Lichess puzzle CSV convention.
- Validation split is by puzzle, not by individual position, to avoid positions from the same puzzle leaking between training and validation.

The script reuses the existing 18-plane encoder, move indexing, and ChessNet architecture. It does not change the model architecture or replay buffer. It starts from \`models/human_pretrained.pt\` by default, falling back to the configured model checkpoint if that file is absent, and writes a separate \`models/mate_pretrained.pt\`. It refuses to overwrite protected source checkpoints. The output can be loaded in the Play GUI with **Load checkpoint…**.

### Dataset
Download the Lichess puzzle database CSV from:
https://database.lichess.org/#puzzles

Place the decompressed CSV at:
\`data/puzzles/lichess_db_puzzle.csv\`

The compressed database is very large; extract the CSV before running the script and ensure sufficient disk space. The script streams rows from the CSV and defaults to processing at most 50,000 rows (not necessarily 50,000 usable mate puzzles).

### Run
From PowerShell in the project directory:

\`\`\`powershell
New-Item -ItemType Directory -Force .\\data\\puzzles
python .\\pretrain_checkmates.py --puzzles .\\data\\puzzles\\lichess_db_puzzle.csv --base-checkpoint models\\human_pretrained.pt --output models\\mate_pretrained.pt --max-puzzles 50000 --epochs 4
\`\`\`

For a smaller first smoke test:

\`\`\`powershell
python .\\pretrain_checkmates.py --max-puzzles 2000 --epochs 1 --output models\\mate_test.pt
\`\`\`

If the CSV's FEN is already after the opponent setup move, pass \`--no-skip-setup-move\`. The default follows the standard Lichess CSV convention and skips the first listed UCI move. The script reports validation top-1 accuracy on held-out puzzles.

### Safety and limitations
- Existing \`latest.pt\`, \`human_pretrained.pt\`, merged checkpoints, historical checkpoints, and replay data are not modified.
- Default output is a new file, \`models/mate_pretrained.pt\`; if it exists, choose another output or explicitly use \`--overwrite\` only for that separate output.
- This is supervised puzzle imitation, not search. It teaches candidate mating moves from puzzle solutions, but does not itself guarantee forced-mate calculation or improve general chess strength.
- The first run should be a small smoke test. Check the number of accepted puzzles and validation accuracy before scaling up.
- The script and menu integration are committed to GitHub but have not been run on the user's Windows PC.

The main menu now includes **Pretrain checkmates from puzzles** under Train & Evaluate.


## 18. Multi-checkpoint weighted blending (2026-11-14)

Expanded checkpoint blending from a fixed two-model operation to a selectable weighted blend of any two or more compatible checkpoints.

### Files updated
- \`merge_checkpoints.py\` — accepts repeated \`--model PATH=PERCENT\` arguments and blends any number of model state dictionaries.
- \`main_menu.py\` — **Combine selected models (weighted)** opens a scrollable checkpoint selector with checkboxes and percentage sliders.
- \`PROJECT_CONTEXT.md\` — records the new interface, CLI, safety rules, and limitations.

### Menu workflow
1. The dialog scans \`models/*.pt\` and lists the available checkpoints.
2. Select any two or more checkpoints.
3. Adjust each selected model's percentage with its slider.
4. The selected percentages must total exactly 100%; the dialog displays the running total and whether it is ready.
5. Choose a new output filename. The output is saved as a separate checkpoint.

The initial slider values favor a three-way blend when \`latest.pt\`, \`human_pretrained.pt\`, and \`mate_pretrained.pt\` are present (34%, 33%, 33%). Other discovered checkpoints are available too. Unselected models are ignored.

### CLI examples

\`\`\`powershell
# Equal three-way blend
python .\\merge_checkpoints.py --model models\\latest.pt=34 --model models\\human_pretrained.pt=33 --model models\\mate_pretrained.pt=33 --output models\\combined_three.pt

# Blend four checkpoints
python .\\merge_checkpoints.py --model models\\latest.pt=40 --model models\\human_pretrained.pt=25 --model models\\mate_pretrained.pt=25 --model models\\cycle_0010.pt=10 --output models\\combined_four.pt
\`\`\`

Weights must add to 100%. At least two selected models must have a non-zero weight. All selected checkpoints must contain compatible model-state keys, tensor shapes, and dtypes. Floating/complex tensors are blended by weighted sum. Integer and boolean buffers (including BatchNorm counters) are copied from the highest-weight checkpoint. Optimizer states are not merged.

### Compatibility and data safety
- The old two-model command-line arguments remain supported for compatibility.
- The output must be distinct from every source checkpoint; existing output files are protected unless \`--overwrite\` is explicitly passed.
- No source checkpoint, historical checkpoint, PGN archive, or replay buffer is modified.
- This is parameter interpolation, not a true knowledge-preserving merge. Blending checkpoints from different training trajectories can damage performance; test the output against each source before using it.
- New implementation has been committed to GitHub but has not been run on the user's Windows PC.

## 18. Unified-model training preparation — 2026-11-14

The goal is to train one compatible `ChessNet` on a balanced mixture of chess knowledge sources, rather than blending independently trained checkpoint weights.

### Local data inventory reported by the user

- Master-game PGNs under `data/pgn/`: Anand, Carlsen, Caruana, Fischer, WorldChamp2023, and WorldChamp2024.
- Lichess puzzle CSV: `data/puzzles/lichess_db_puzzle.csv` (approximately 1.18 GB).
- Replay candidates:
  - `data/replay_buffer.npz`
  - `data/replay_buffer_eval_v2.npz`
  - `data/replay_buffer_move_game_v3.npz`

### Checkpoints reported locally

- `models/latest.pt`
- `models/human_pretrained.pt`
- `models/mate_pretrained.pt`
- `models/combined.pt`
- `models/cycle_0527.pt`
- `models/mate_test.pt`

The user reports completing human-PGN pretraining and checkmate-puzzle pretraining. The checkmate checkpoint reports a validation top-1 move accuracy of approximately 18.45% over 2,591 validation positions. Treat this as a baseline metric only; it does not establish reliable mate-solving ability.

### Replay validation results

The user ran a read-only NumPy inspection of `data/replay_buffer.npz`. It contains:

- `positions`: shape `(22422, 18, 8, 8)`, uint8
- `teacher`: shape `(22422,)`, int32
- `values`: shape `(22422,)`, float32
- `chosen`: shape `(22422,)`, int32
- `rewards`: shape `(22422,)`, float32
- `legal_offsets`: shape `(22423,)`, int32
- `legal_actions`: shape `(696185,)`, int32

The follow-up validation reported:
- Teacher action range: 1–19,967; no action outside 0–20,479.
- Chosen action range: -1–17,221; -1 is the expected sentinel for no selected action.
- Value and reward ranges: both -1.0 to 1.0.
- No empty legal-action lists.
- No invalid teacher, chosen, or legal-action indices.
- No non-finite value targets.

This is a strong initial indication that `replay_buffer.npz` is structurally usable. Before training, a loader should still verify exact array-length agreement, monotonic offsets, final offset equal to the legal-action array length, and that each teacher action occurs in its position's legal-action slice.

Two other replay files failed read-only loading:
- `replay_buffer_eval_v2.npz`: EOF while reading compressed array data.
- `replay_buffer_move_game_v3.npz`: decompression error / invalid block type.

Do not delete, overwrite, repair in place, or use these two files as training inputs unless separately backed up and successfully validated. Preserve all replay files regardless.

### Unified training design

The intended unified model will start from `models/human_pretrained.pt`, not from a weighted average of checkpoints. It should train the existing architecture with balanced/interleaved batches from:
1. Human/master PGN positions.
2. Lichess checkmate puzzles (mate-in-1/2/3).
3. Other tactical puzzle themes.
4. Endgame puzzle themes.
5. The validated `data/replay_buffer.npz`, if all structural and semantic checks pass.

The first output should be a new file, `models/unified.pt`. It must not overwrite any source checkpoint, the main `latest.pt`, or any replay file. Keep the existing 18-plane encoder, 20,480-action move encoding, and network architecture unchanged.

A mixed-source trainer should prevent the large puzzle CSV from overwhelming the smaller sources by sampling a controlled amount from each category per epoch/batch. It should report validation results separately by source so gains in one area do not hide forgetting in another. Puzzle validation should split by puzzle identity, not by individual positions from the same puzzle.

The unified trainer has not yet been implemented or run. The next development step is to add this separate trainer and update this context again with its exact command-line options. No local files have been changed by the GitHub context update.


## 19. Unified mixed-source trainer added (2026-11-14)

Added train_unified.py in commit 3915283bb6c04a8accb84ec26f6c7057337204e9.

### Purpose and method
This is single-model fine-tuning, not checkpoint averaging. It initializes ChessNet from models/human_pretrained.pt and trains the same model on balanced source-specific batches. It preserves the 18-plane encoder, 20,480-action policy, and value head.

Supported sources:
- human_pgn: completed master-game PGNs; move policy targets plus game-result value targets. Validation is split by game.
- mate_puzzles: Lichess mate-in-1 through mate-in-5 themes; accepts complete solution lines ending in checkmate.
- tactics_puzzles: recognized tactical themes such as forks, pins, skewers, discovered attacks, sacrifices, and defensive moves.
- endgame_puzzles: recognized endgame themes such as rook, bishop, knight, pawn, and queen endgames.
- replay: data/replay_buffer.npz; policy and value targets are used. The loader rechecks array dimensions, offset structure, finite values, action ranges, and that every teacher move is legal in its position.

Puzzle samples are split by puzzle ID, so positions from one puzzle cannot leak across train and validation. Puzzle positions train the policy head only; they do not invent value labels. PGN and replay sources train both policy and value heads.

Each epoch samples the configured number of positions from each available source, rather than concatenating datasets and letting the large puzzle database dominate. Validation top-1 accuracy is reported separately for each source; value MAE is reported where real value targets exist.

### Default run

From the project root in PowerShell:

```powershell
python .\train_unified.py
```

Defaults: base models/human_pretrained.pt; output models/unified.pt; PGN directory data/pgn; puzzle CSV data/puzzles/lichess_db_puzzle.csv; replay data/replay_buffer.npz (skipped with --no-replay); scan up to 100,000 puzzle CSV rows and 100,000 PGN positions; sample 10,000 positions per source per epoch; 3 epochs; batch size 128; learning rate 0.00001; 10% validation split.

For a smaller first smoke test:

```powershell
python .\train_unified.py --puzzle-rows 5000 --max-pgn-positions 10000 --samples-per-source 1000 --epochs 1 --output models\unified_test.pt
```

If the output already exists, choose another output name or pass --overwrite only when intentionally replacing that output. The script protects latest.pt, human_pretrained.pt, and mate_pretrained.pt as output targets. It never writes to input datasets.

### Important interpretation and limitations
- This is the first implementation and has been committed but has NOT yet been run or syntax-tested on the user's Windows PC. Run the small smoke test before the default run and share any traceback.
- Puzzle category sets are explicit theme allowlists. Unlisted Lichess themes are ignored; multi-theme puzzles are assigned by priority mate, then endgame, then tactics.
- Tactic/endgame puzzle lines are policy demonstrations; only mate-themed lines are required to end in checkmate.
- Equal sample counts balance example counts, not difficulty or information content.
- Validation top-1 is exact move imitation under a legal-move mask. It is not a chess-strength rating and does not prove independent tactical or forced-mate calculation.
- The trainer does not use replay_buffer_eval_v2.npz or replay_buffer_move_game_v3.npz because both failed earlier read-only decompression checks. Preserve them untouched.
- Existing checkpoints, PGN archives, puzzle CSV, and replay files remain preserved. The new output is separate at models/unified.pt.

### Next steps
1. Safely sync train_unified.py and this context update to the local checkout. Preserve the previous local stash and untracked data/evaluation file; inspect git status --short --branch and git stash list before pulling. Do not use git reset --hard, git clean, or blindly pop the stash.
2. Run the small smoke test and confirm each expected source loads and models/unified_test.pt is saved.
3. If successful, run the default command to create models/unified.pt.
4. Benchmark unified.pt against human_pretrained.pt, mate_pretrained.pt, and latest.pt with identical arena settings before choosing a default model.
