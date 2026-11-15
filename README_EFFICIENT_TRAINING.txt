This package retains the efficient CPU training configuration.
For the new separate individual-move and whole-game reward behavior, see
README_MOVE_AND_GAME_REWARDS.txt.

The model resumes from models/latest.pt. The new replay buffer path is
 data/replay_buffer_move_game_v3.npz; the older replay buffer is left untouched.
