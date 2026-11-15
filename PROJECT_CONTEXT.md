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

The proposed initial collection from PGN Mentor includes games by Carlsen, Caruana, Anand, and Fischer, with selected World Championship collections as optional additions. The user has now reported that PGN training has been completed locally. Exact local PGN filenames/counts and the saved output path have not yet been independently verified.

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

The script was committed to GitHub. The user has now reported that PGN pretraining completed locally. Treat this as user-reported completion; the assistant has not independently inspected the local checkpoint or training log.

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
- [x] Add a separate Stockfish evaluation script for comparing the original and PGN-pretrained checkpoints.
- [ ] Run `evaluate_stockfish.py` locally and inspect the results.
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

1. Finish Git synchronization safely. Check whether the previous stash/pull completed.
2. Run `python .\\evaluate_stockfish.py --games-per-model 10` locally to compare `models/latest.pt` against `models/human_pretrained.pt`.
3. Review `evaluation_results.csv`; if results are close or noisy, repeat with 30–50 games per model.
4. Confirm the baseline checkpoint is still the original pre-PGN model and that the replay buffer remains unchanged.
5. Only if the pretrained checkpoint performs acceptably, decide whether to use it as the starting point for continued Stockfish-guided training.
6. After this is validated, prioritize tactical puzzle training and then endgame knowledge.

## 13. User preferences and implementation rules

- Provide complete, usable code and exact Windows PowerShell commands.
- Prefer making and committing repository changes directly when GitHub access is available.
- Explain what changed and what the user must do locally.
- Avoid unnecessary back-and-forth; ask only for missing information needed to proceed.
- Preserve all existing model checkpoints and replay data by default.
- Do not claim code has been tested locally unless the user has actually run it or a suitable test environment was used.
- Do not claim GitHub edits have been pulled onto the user's PC.
- Whenever the project changes, update this `PROJECT_CONTEXT.md` so it accurately records the current state, what is implemented, what is untested, and the next steps.
