StockfishNN low-CPL reward update

Replace train.py with this version. It retains the evaluation-target training,
checkpoints, replay format, and launchers from the previous reward-fix package.

Reward change:
- Approximate move CPL = max(0, eval_before_from_NN_perspective -
  eval_after_move_from_NN_perspective).
- CPL reward = 1 - 2*tanh(CPL/300).
- The existing uncompensated material-loss penalty is retained.
- Reward is clamped to [-1, +1].

Interpretation:
- 0 CPL: +1.00 before material penalty
- 50 CPL: about +0.67
- 100 CPL: about +0.36
- 200 CPL: about -0.17
- 300 CPL: about -0.52
- 500 CPL: about -0.87

This is an efficient approximate CPL signal using the existing Stockfish
evaluations. It is not a separate full best-move-vs-played-move analysis.
Use benchmark win/draw/loss and average CPL to judge playing strength, not
the reward number alone.

Installation:
1. Stop training and back up the project.
2. Extract this ZIP into the StockfishNN folder.
3. Replace train.py only.
4. Keep models and data folders.
5. Start training. Existing model weights and replay data remain compatible.
