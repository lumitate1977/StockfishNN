\# StockfishNN — Project Context



\## 1. Project Overview



\*\*Project name:\*\* StockfishNN  

\*\*Location:\*\* `C:\\Users\\thoma\\Downloads\\StockfishNN`  

\*\*Purpose:\*\* Build a chess neural network using PyTorch that learns to play chess through Stockfish supervision, reinforcement-style rewards, replay training, and eventually self-play.



The goal is to progressively improve the neural network until it can play chess independently and become stronger through continued training.



\## 2. Hardware and Environment



\- OS: Windows

\- CPU: AMD Ryzen 7 5800X

\- GPU: AMD Radeon RX 6600 8GB

\- RAM: 32GB DDR4

\- Python: 3.13

\- Framework: PyTorch

\- Chess library: python-chess

\- Training backend: Primarily CPU

\- Stockfish: External UCI engine



The RX 6600 does not normally provide CUDA support for standard PyTorch builds, so the current training setup uses CPU unless another supported backend is configured.



\## 3. Existing Project Files



Important files include:



\- `train.py` — Main training system.

\- `chess\_ai.py` — Neural network architecture, board encoding, move selection, and model loading.

\- `game\_archive.py` — Archives training games and calculates game metrics.

\- `config.json` — Configuration, including model path and Stockfish path.

\- `models/latest.pt` — Latest neural network checkpoint.

\- `models/cycle\_XXXX.pt` — Historical cycle checkpoints.

\- `data/replay\_buffer\_move\_game\_v3.npz` — Replay buffer.

\- `training\_log\_eval\_v2.csv` — Training statistics.

\- `play\_gui.py` — GUI for playing against the neural network and reviewing games.



Other `.bat` files may be present for launching the different project features.



\## 4. Neural Network Architecture



The current `ChessNet` uses:



\- Input: 18 board planes.

\- Convolutional stem: 48 channels.

\- Two residual blocks.

\- Shared fully connected layer: 256 units.

\- Policy head: 20,480 possible move actions.

\- Value head: predicts a position evaluation in the range \[-1, 1].



\### Board encoding



The 18 input planes represent:



\- 12 piece planes (six piece types for each color).

\- Four castling-right planes.

\- One side-to-move plane.

\- One en-passant plane.



\### Move encoding



`ACTION\_SIZE = 5 \* 64 \* 64 = 20480`



The five promotion slots represent:



1\. Normal move.

2\. Knight promotion.

3\. Bishop promotion.

4\. Rook promotion.

5\. Queen promotion.



Illegal moves are masked before policy selection.



\## 5. Current Training System



The training system uses:



\- Stockfish as the teacher and opponent.

\- Neural-network move selection.

\- Stockfish position evaluations.

\- Individual move-quality rewards.

\- Whole-game rewards.

\- Replay-buffer training.

\- Policy and value losses.

\- Periodic checkpoint saving.

\- Optional benchmark games.



\### Current default settings



| Setting | Default |

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



These are command-line defaults, not necessarily the settings used in every training run.



\## 6. Reward System



The current implementation combines two reward types:



\- \*\*60% individual move-quality reward\*\*

\- \*\*40% whole-game reward\*\*



The console message currently incorrectly says 75%/25%. The actual implementation in `train\_epoch()` uses 60%/40%.



\### Individual move reward



The system estimates move quality from Stockfish evaluation changes.



It calculates an approximate centipawn loss:



`move\_cpl = max(0, before\_cp - after\_for\_mover\_cp)`



The reward is:



`cpl\_reward = 1 - 2 \* tanh(move\_cpl / 300)`



This means:



\- Good moves receive positive rewards.

\- Moves with small evaluation drops receive smaller rewards.

\- Major mistakes receive negative rewards.



An additional material-loss penalty is applied when material loss is not already explained by the evaluation change.



\*\*Important:\*\* This is an approximate evaluation-drop reward, not a full Stockfish best-move-versus-played-move CPL calculation.



\### Whole-game reward



Completed games use their actual result:



\- Win: +1

\- Draw: 0

\- Loss: -1



Unfinished games use the final Stockfish evaluation rather than being treated as draws.



\### Training loss



The current combined loss is:



`total\_loss = supervised\_loss + 0.5 \* value\_loss + 0.05 \* reinforce\_loss`



The supervised policy loss teaches the NN to imitate Stockfish's selected moves.



The value loss teaches the network to estimate position quality.



The reinforcement-style loss uses the NN's selected moves and their rewards.



\## 7. Replay Buffer



The replay buffer stores:



\- Encoded board positions.

\- Stockfish teacher actions.

\- Value targets.

\- Legal actions.

\- NN-selected actions.

\- Individual move rewards.

\- Whole-game rewards.



The existing buffer is:



`data/replay\_buffer\_move\_game\_v3.npz`



It supports older buffer formats that may not contain the newer reward arrays.



\*\*Do not delete or reset this buffer during upgrades.\*\*



It contains accumulated training experience.



\## 8. Checkpoints and Resuming



The existing checkpoint format contains:



\- `model\_state`

\- `optimizer\_state`

\- `metadata`



The main model is:



`models/latest.pt`



Historical checkpoints are:



`models/cycle\_XXXX.pt`



Training automatically resumes from `models/latest.pt` unless `--fresh` is used.



The `--fresh` argument starts a new model and empty replay buffer.



\*\*Never use `--fresh` unless the user explicitly wants to restart training.\*\*



Preserve the existing model and replay buffer during all upgrades.



\## 9. Current Training Loop



For every cycle:



1\. Generate training games.

2\. Randomly assign the NN to White or Black.

3\. Use Stockfish to select teacher moves and evaluate positions.

4\. Let the NN choose moves on its turns.

5\. Calculate individual move rewards.

6\. Calculate game-level rewards.

7\. Archive completed games.

8\. Add collected positions to the replay buffer.

9\. Train the network using replay samples.

10\. Run a benchmark if the configured cycle interval is reached.

11\. Save `latest.pt`.

12\. Save a numbered cycle checkpoint.

13\. Save the replay buffer.

14\. Write statistics to the CSV log.



\## 10. Planned Training Improvements



The next major upgrade should improve the quality and variety of training opponents.



\### A. Mixed Stockfish difficulties



Instead of always playing against one Stockfish skill level, rotate between different strengths.



Proposed default pool:



\- Skill 5

\- Skill 8

\- Skill 12



The difficulty should be selected per game.



The benchmark should continue using a fixed difficulty so that results remain comparable across cycles.



\### B. Self-play against older neural networks



Load several previous cycle checkpoints as frozen opponent models.



For example:



\- Current NN plays against `cycle\_0010.pt`.

\- Current NN plays against `cycle\_0011.pt`.

\- Current NN plays against `cycle\_0012.pt`.



The exact checkpoints should depend on which ones exist.



Proposed initial self-play probability: 20%.



Stockfish should continue evaluating positions and supplying teacher moves, even when an older NN controls the opposing side.



The older models must remain frozen and must not be modified during training.



\### C. Human-game pretraining



Add a separate script called:



`pretrain\_human.py`



It should:



1\. Read standard chess PGN files.

2\. Encode positions using the existing 18-plane encoder.

3\. Convert human moves into the existing action format.

4\. Mask illegal moves.

5\. Train the policy head through supervised learning.

6\. Save a separate checkpoint.



Suggested output:



`models/human\_pretrained.pt`



It must load the existing model before pretraining and must not overwrite `models/latest.pt` automatically.



A manageable Lichess PGN sample should be used initially.



\### D. Better benchmarking



Use fixed Stockfish strength for benchmark games.



Alternate NN colors.



Track:



\- Wins.

\- Draws.

\- Losses.

\- Unfinished games.

\- Win rate.

\- Average centipawn loss.

\- Performance over time.



Benchmark results should be logged separately from ordinary training games.



\## 11. Important Implementation Rules



When modifying the project:



1\. Preserve the existing `ChessNet` architecture unless an architecture change is specifically requested.

2\. Preserve the existing board encoding and move encoding.

3\. Preserve checkpoint compatibility.

4\. Preserve the replay buffer and its stored experience.

5\. Do not reset training automatically.

6\. Keep the 60/40 reward system unless the user requests a change.

7\. Keep Stockfish evaluation active during self-play.

8\. Do not let older opponent networks receive gradient updates.

9\. Keep benchmark difficulty fixed.

10\. Prefer CPU-efficient implementations.

11\. Avoid unnecessary simultaneous Stockfish processes.

12\. Back up files before applying large changes.



\## 12. Current Upgrade Status



The user has provided the relevant source code sections from `train.py`:



\- Imports and data structures.

\- Replay buffer.

\- Stockfish configuration.

\- `play\_training\_game()`.

\- Reward calculations.

\- `build\_training\_arrays()`.

\- `train\_epoch()`.

\- Checkpoint saving.

\- Argument parsing and model initialization.

\- Complete training loop.



The full `chess\_ai.py` was also previously provided.



The proposed upgrade has been designed conceptually, but \*\*the actual patch and ZIP have not yet been generated or installed\*\*.



Do not claim the upgrade is installed.



The intended upgrade includes:



\- Mixed Stockfish difficulty.

\- Self-play against older checkpoints.

\- Human PGN pretraining.

\- Corrected 60/40 reward console message.

\- Fixed-strength benchmarks.

\- Preservation of existing model and replay data.



\## 13. User Preferences



The user prefers:



\- Complete working code rather than vague suggestions.

\- Ready-to-use replacement files or ZIP archives when possible.

\- Exact PowerShell commands.

\- Clear explanations of what changed.

\- Minimal unnecessary back-and-forth.

\- Preserving existing work and training progress.



When asking for code, request only the specific missing section.



Do not repeatedly ask for code that has already been provided.



\## 14. Immediate Next Task



Continue the planned training upgrade.



First, implement and test the opponent-pool upgrade:



\- Mixed Stockfish strengths.

\- Older neural-network opponents.

\- Fixed-strength benchmark.

\- Preserve all current training functionality.



Then add human PGN pretraining as a separate script.



Before modifying anything, make backups of `train.py`, `models/latest.pt`, and the replay buffer.



After implementation, provide exact installation and launch commands for Windows PowerShell.

## 15. Human PGN Pretraining — Implemented

The repository now includes `pretrain_human.py`, a separate supervised pretraining program for human/master games.

### Data location and formats

- Put downloaded game files in `data/pgn/`.
- Supported inputs: `.pgn` and gzip-compressed `.pgn.gz`.
- The script searches the folder recursively.
- It uses each game's mainline only; side variations are ignored.
- Games with an unfinished result (`*`) are skipped.
- Optional `--min-elo` requires both players to have Elo ratings at or above the chosen threshold.

### What it teaches

For every mainline position, the script:
1. Encodes the board with the existing 18-plane encoder.
2. Uses the human move as the policy target.
3. Builds the legal-move set for that position and masks illegal actions in the policy loss.
4. Uses the final game result as a value target from the side-to-move perspective.
5. Fine-tunes the existing `ChessNet` architecture.

This is supervised imitation of recorded games, not a replacement for Stockfish evaluation, replay training, or self-play. Game-outcome value targets are comparatively coarse and do not provide Stockfish's position-level evaluation.

### Safe checkpoint behavior

- Default starting checkpoint: `config.json`'s `model_path` (currently `models/latest.pt`).
- Default output: `models/human_pretrained.pt`.
- The script refuses to use the same path for input and output.
- It refuses to overwrite an existing output unless `--overwrite` is supplied.
- It does not write to the main checkpoint or replay buffer.
- The user should evaluate the separate pretrained checkpoint before choosing to promote it.

### Example commands (PowerShell)

From the StockfishNN project directory:

```powershell
New-Item -ItemType Directory -Force .\data\pgn
# Place downloaded .pgn or .pgn.gz files in .\data\pgn
python .\pretrain_human.py --pgn-dir .\data\pgn --epochs 3 --max-positions 100000
```

Optional stronger-game filter:

```powershell
python .\pretrain_human.py --pgn-dir .\data\pgn --min-elo 2200 --epochs 3
```

If `models/human_pretrained.pt` already exists, choose a different `--output` path or explicitly pass `--overwrite`.

### Current status

- [x] Add separate PGN pretraining script.
- [x] Preserve the current model architecture and checkpoint format.
- [x] Mask illegal moves during supervised policy training.
- [x] Support PGN and gzip-compressed PGN input.
- [x] Save to a separate checkpoint and protect `models/latest.pt`.
- [ ] Add/download a suitable master-game PGN collection.
- [ ] Run a small smoke test on the user's local machine.
- [ ] Evaluate the pretrained checkpoint against the unmodified model.
- [ ] Decide whether to promote the pretrained checkpoint for continued Stockfish/self-play training.

The GitHub source update does not download PGNs, run training on the user's PC, or change local checkpoints/replay data. Those remain local operations.
