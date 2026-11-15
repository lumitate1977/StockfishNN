StockfishNN - Separate Move + Game Rewards
==========================================

WHAT CHANGED
------------
The NN now receives two distinct reward signals for every move it chose:

1. Individual move reward (75% of the combined policy reward)
   - Based on the existing approximate centipawn-loss signal.
   - Good, low-CPL moves receive positive feedback.
   - Mistakes and uncompensated material losses receive lower/negative feedback.

2. Whole-game reward (25% of the combined policy reward)
   - Every NN move in a completed win receives +1.
   - Every NN move in a draw receives 0.
   - Every NN move in a loss receives -1.
   - If the game reaches the training ply limit instead of finishing, the final
     Stockfish evaluation is converted into a bounded game reward. It is NOT
     incorrectly treated as a draw.

The game reward is only attached to positions where the NN actually selected
and played a move. Stockfish's moves do not receive the NN's policy reward.
The value head continues to learn from Stockfish evaluations and game outcomes.

INSTALL
-------
1. Stop training.
2. Make a backup copy of your StockfishNN folder (or at least models and data).
3. Extract this ZIP into C:\Users\thoma\Downloads\StockfishNN and allow
   train.py to be replaced.
4. Keep your existing models, data, games, config.json, and GUI files.
5. Run train.bat as usual.

MODEL AND REPLAY DATA
--------------------
- models/latest.pt is still resumed. Your trained network weights are kept.
- The new reward format uses data/replay_buffer_move_game_v3.npz.
- The previous replay buffer is NOT deleted or overwritten. The new buffer starts
  separately so old rewards are not mixed with the new two-part reward labels.
- The new replay buffer fills as new training games are played.
- Do not use --fresh unless you intentionally want to reset the model as well.

WHAT TO WATCH
-------------
Each cycle now prints separate mean move reward and mean game reward.
Also watch the W/D/L results, average NN CPL, and fixed benchmark W/D/L.
A positive reward is not proof of stronger play. Compare benchmarks over many
cycles because individual games are noisy.

LIMITATION
----------
The move reward still uses the existing fast approximate CPL estimate based on
Stockfish evaluations before and immediately after the played move. It is not
an exact independent best-move-vs-played-move CPL search. This keeps the added
reward mechanism efficient. The code was syntax-checked, but must still be run
in the user's Windows environment to validate a complete training cycle.
