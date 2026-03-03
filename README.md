# RL-Coursework

Reinforcement Learning Coursework: Gomoku (15×15) with an AlphaZero-style self-play trained agent.

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Pygame (for the game UI only; training runs headless)

Install dependencies (example with pip):

```bash
pip install torch numpy pygame
```

## Running the game

Play against the AlphaZero bot in a 15×15 Gomoku window:

```bash
python main.py
```

The game loads the trained model from `weights/best.pt` if present; otherwise the bot uses an untrained (random-initialized) network.

## Training (no UI)

Training is self-play only and does not use Pygame. Run from the project root:

```bash
python train.py
```

Optional arguments:

- `--board_size` (default: 15)
- `--num_simulations` (default: 50) – MCTS simulations per move during self-play
- `--games_per_iteration` (default: 100)
- `--batch_size` (default: 64)
- `--learning_rate` (default: 1e-4)
- `--iterations` (default: 100)
- `--train_epochs` (default: 3) – epochs per iteration over the replay buffer
- `--checkpoint_dir` (default: `weights`)
- `--save_best_path` (default: `weights/best.pt`)
- `--resume` – path to a checkpoint to resume from
- `--device` – e.g. `cuda` or `cpu`
- `--value_coef` (default: 1.0) – weight of value loss vs policy loss
- `--seed` – random seed
- `--no-compile` – disable `torch.compile` for the model (use if you see errors or PyTorch is below 2.0)
- `--mcts_batch_size` (default: 32) – batch size for MCTS leaf evaluation (larger = fewer NN calls, more memory)

**Note:** With PyTorch 2+, the model is compiled by default for faster inference. The first run after starting training may be slower due to tracing; later runs are faster. Use `--no-compile` to disable compilation.

Checkpoints are written to `checkpoint_dir` every 5 iterations; the best model (by loss) is saved to `save_best_path`.

## Model weights

- **Default path**: defined in `src/model_loader.py` as `DEFAULT_WEIGHTS_PATH` (e.g. `weights/best.pt`). Save/load helpers: `save_weights(model, path)`, `load_weights(model, path, device)`.
- **Training output**: `weights/checkpoint_*.pt` and `weights/best.pt`.
- **Game UI**: uses `DEFAULT_WEIGHTS_PATH` when using the AlphaZero bot.

## Standardized API (cross-group testing)

Create one bot and call `bot.predict()` for each board state (no reinitialisation):

```python
from src.Bots.alpha_zero_transform import Bot
from src.model_loader import DEFAULT_WEIGHTS_PATH

bot = Bot(weights_path=DEFAULT_WEIGHTS_PATH)  # or weights_path="path/to/weights.pt"
# For each position:
move = bot.predict(board_state, current_player=0)  # (x, y); current_player optional if inferred
```

One-off use: `predict(board_state, current_player=None, weights_path=..., **kwargs)` from `src.Bots.alpha_zero_transform`, or pass a pre-created `bot=...` to avoid creating a new one.

## Project structure

- `main.py` – Pygame game loop; launches the playable scene.
- `train.py` – Self-play and training script (no UI).
- `src/game_logic.py` – Gomoku game logic and bot loading.
- `src/gomoku_game.py` – Board helpers and win/draw detection.
- `src/gomoku_utils.py` – Board-to-3-plane encoding for the network.
- `src/mcts.py` – MCTS and policy target for training.
- `src/Bots/base_bot.py` – Abstract bot interface (`move(game_state)`).
- `src/Bots/alpha_zero_transform.py` – Transformer policy/value model and AlphaZero bot; `bot.predict()` and module-level `predict()` API.
- `src/model_loader.py` – Default weights path, `save_weights()`, `load_weights()`.
- `src/Bots/random.py` – Random-move bot.
- `src/Scenes/` – Pygame scenes (game UI).
