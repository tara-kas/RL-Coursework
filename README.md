# RL-Coursework

![](https://github.com/tara-kas/RL-Coursework/assets/gameplay.gif)

Reinforcement Learning Coursework: Gomoku (15×15) with multiple agents (AlphaZero, PPO, DQN, heuristic, random).

## Recent updates (main branch)

- `main.py` currently uses bot-module args (`--bot_file`, `--bot_kwargs`) and defaults to a **player vs PPO** setup.
- PPO checkpoints in root:
  - `p0_ppo.pt`
  - `p1_ppo.pt`
- PPO wrapper now degrades gracefully: if PPO dependencies/checkpoint are missing, it falls back to a tactical bot so the UI can still run outside a shared venv.

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Pygame (for the game UI only; training runs headless)

Optional packages for PPO backend:
- omegaconf
- tensordict
- torchrl
- hydra-core

Install dependencies (example with pip):

```bash
pip install torch numpy pygame omegaconf tensordict torchrl hydra-core
```

## Running the game

Current default UI run (player vs PPO):

```bash
python main.py
```

Explicit PPO run:

```bash
python main.py \
  --board_size 15 \
  --bot_file ppo_gomoku_model_15x15 \
  --bot_name PPO \
  --bot_kwargs '{"weights_path":"p0_ppo.pt","device":"cpu","deterministic":true}'
```

Current AlphaZero UI run:

```bash
python main.py --board_size 9 --bot_file alpha_zero_resnet --bot_name "AlphaZero ResNet" --weights_path best_weights/alphazero_best.pt --bot_kwargs "{\"num_simulations\": 200}"
```

Current Transformer UI run:

```bash
python main.py --board_size 9 --bot_file alpha_zero_transformer --bot_name "AlphaZero Transformer" --weights_path temp_weights/checkpoint_120.pt --bot_kwargs "{\"num_simulations\": 200}"
```

Legacy/older-branch AlphaZero command:

```bash
python main.py --board_size 9 --agent_type alphazero-resnet --weights weights/checkpoint_165.pt
```

## Training (no UI)

Training uses self-play plus optional games against a heuristic tactical bot and past checkpoints (league). Run from the project root:

```bash
python train.py
```

Common arguments:

- `--board_size` (default: 15)
- `--agent_type` (`dqn`, `alphazero`, `alphazero-resnet`, `alphazero-transformer`, `hybrid`, `alphazero-hybrid`)
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
- `--c_puct` (default: 1.5) – MCTS exploration constant; higher values encourage more exploration
- `--self_play_temp` (default: 1.0) – temperature for move sampling in the first `--temp_moves` of each game (0 = argmax)
- `--temp_moves` (default: 30) – number of moves per game with temperature; after that, argmax
- `--league_prob` (default: 0.25) – probability each game is played vs a random past checkpoint
- `--heuristic_prob` (default: 0.2) – probability each game is played vs the heuristic tactical bot (win/block-4)
- `--league_pool_size` (default: 5) – max number of past checkpoints kept in the league pool
- `--amp` – use FP16 autocast in MCTS for faster inference on GPU
- `--num_workers` (default: 1) – number of parallel self-play workers (1 = no parallelism; workers run on CPU by default)
- `--worker_device` (default: cpu) – device for parallel workers; main process keeps GPU for training
- `--az_replay_buffer_size` – replay buffer capacity for AlphaZero and hybrid training
- `--az_best_by` – AlphaZero/hybrid best-model criterion (`loss` or `heuristic`)
- `--best_by` – DQN best-model criterion (`loss` or `heuristic`)
- `--heuristic_win_bonus` – DQN-only reward shaping for wins vs heuristic
- `--heuristic_prob_start`, `--heuristic_prob_decay_iters` – DQN curriculum controls

**Note:** With PyTorch 2+, the model is compiled by default for faster inference. The first run after starting training may be slower due to tracing; later runs are faster. Use `--no-compile` to disable compilation.

Training uses temperature and opponent diversity (heuristic + league) to improve tactics and reduce policy collapse. Checkpoints are saved regularly and can be reused for resume or league play.

## Model weights

- **Default path**: defined in `src/model_loader.py` as `DEFAULT_WEIGHTS_PATH` (e.g. `weights/best.pt`). Save/load helpers: `save_weights(model, path)`, `load_weights(model, path, device)`.
- **Training output**: `weights/checkpoint_*.pt` and `weights/best.pt`.
- **Game UI**: uses `DEFAULT_WEIGHTS_PATH` when using the AlphaZero bot.

## Evaluation

Run standalone evaluation of a model vs random, heuristic, DQN, or AlphaZero-family opponents:

```bash
# DQN vs random (1000 games)
uv run python evaluation.py --model weights/dqn_best.pt --agent_type dqn --opponent random --num_games 1000 --board_size 9

# DQN vs heuristic
uv run python evaluation.py --model weights/dqn_best.pt --agent_type dqn --opponent heuristic --num_games 1000 --board_size 9

# DQN vs AlphaZero ResNet (requires opponent weights)
uv run python evaluation.py --model weights/dqn_best.pt --agent_type dqn --opponent alphazero-resnet --opponent_weights best_weights/alphazero_best.pt --num_games 1000 --board_size 9 --amp

# Hybrid vs DQN
uv run python evaluation.py --model temp_weights/hybrid_checkpoint_120.pt --agent_type alphazero-hybrid --opponent dqn --opponent_weights best_weights/dqn_9_best.pt --num_games 1000 --board_size 9 --amp
```

Options: `--model` or `--weights` (path), `--agent_type` (`dqn`, `alphazero`, `alphazero-resnet`, `alphazero-transformer`, `hybrid`, `alphazero-hybrid`), `--opponent` (`random`, `heuristic`, `dqn`, `alphazero`, `alphazero-resnet`, `alphazero-transformer`, `hybrid`, `alphazero-hybrid`), `--num_games`, `--board_size`, `--opponent_weights` for learned opponents, `--amp`, `--num_simulations`, `--seed`.

## Standardized API (cross-group testing)

Create one bot and call `bot.predict()` for each board state (no reinitialisation):

```python
from src.Bots.alpha_zero_resnet import Bot
from src.model_loader import DEFAULT_WEIGHTS_PATH

bot = Bot(weights_path=DEFAULT_WEIGHTS_PATH)  # or weights_path="path/to/weights.pt"
# For each position:
move = bot.predict(board_state, current_player=0)  # (x, y); current_player optional if inferred
```

You can do the same with `src.Bots.alpha_zero_transformer` or `src.Bots.alpha_zero_hybrid`, or pass a pre-created `bot=...` to avoid creating a new one.

## Project structure

- `main.py` – Pygame game loop; launches the playable scene.
- `train.py` – Self-play and training script (no UI).
- `evaluation.py` – Standalone evaluation for DQN and AlphaZero-family agents.
- `src/game_logic.py` – Gomoku game logic and bot loading.
- `src/gomoku_game.py` – Board helpers and win/draw detection.
- `src/gomoku_utils.py` – Board-to-3-plane encoding for the network.
- `src/mcts.py` – Python MCTS used by AlphaZero-family agents.
- `src/alphazero_buffer.py` – compact replay buffer for AlphaZero and hybrid training.
- `src/Bots/base_bot.py` – Abstract bot interface (`move(game_state)`).
- `src/Bots/alpha_zero_resnet.py` – ResNet AlphaZero model and bot wrapper.
- `src/Bots/alpha_zero_transformer.py` – Transformer AlphaZero model and bot wrapper.
- `src/Bots/alpha_zero_hybrid.py` – hybrid AlphaZero model and bot wrapper.
- `src/Bots/dqn.py` – DQN model, replay buffer, self-play, and training helpers.
- `src/Bots/dqn_bot.py` – DQN bot wrapper for gameplay.
- `src/Bots/ppo_gomoku_model_15x15.py` – PPO UI wrapper.
- `src/Bots/heuristic_tactical.py` – Heuristic tactical bot (win/block-4); used as an opponent during training.
- `src/model_loader.py` – Default weights path, `save_weights()`, `load_weights()`.
- `src/Bots/random.py` – Random-move bot.
- `src/cpp/` – optional C++ MCTS extension sources.
- `src/Scenes/` – Pygame scenes (game UI).

## Useful Commands

### AlphaZero baseline from scratch

```bash
uv run python train.py --iterations 500 --num_simulations 200 --games_per_iteration 150 --learning_rate 2e-4 --value_coef 2.5 --c_puct 2.0 --self_play_temp 1.0 --temp_moves 30 --league_prob 0.25 --heuristic_prob 0.2
```

### AlphaZero resume from checkpoint

```bash
uv run python train.py --resume weights/checkpoint_100.pt --iterations 500 --value_coef 2.0 --num_simulations 100 --games_per_iteration 150 --learning_rate 2e-4
```

### AlphaZero 9x9 heuristic-focused resume

```bash
python train.py --board_size 9 --agent_type alphazero --amp --num_workers 8 --worker_device cuda --mcts_batch_size 128 --batch_size 512 --iterations 300 --games_per_iteration 400 --eval_games 200 --learning_rate 5e-4 --num_simulations 600 --value_coef 1.0 --c_puct 2.0 --self_play_temp 1.0 --temp_moves 30 --league_prob 0.1 --heuristic_prob 0.5 --az_replay_buffer_size 240000 --az_best_by heuristic --az_eval_freq 10 --az_eval_games_best 200 --root_dirichlet_alpha 0.3 --root_dirichlet_epsilon 0.25 --resume weights/checkpoint_500.pt
```

### AlphaZero ResNet on Quadro RTX 6000

```bash
uv run python train.py --device cuda --board_size 9 --agent_type alphazero-resnet --amp --num_workers 12 --worker_device cpu --mcts_batch_size 64 --batch_size 1024 --iterations 300 --games_per_iteration 400 --eval_games 100 --learning_rate 5e-4 --num_simulations 800 --value_coef 1.0 --c_puct 1.5 --self_play_temp 1.0 --temp_moves 12 --league_prob 0.2 --heuristic_prob 0.15 --az_best_by heuristic --az_eval_freq 10 --az_eval_games_best 100 --root_dirichlet_alpha 0.3 --root_dirichlet_epsilon 0.25 --resume weights/checkpoint_115.pt
```

### AlphaZero settings that previously collapsed policy

```bash
uv run python train.py --board_size 9 --agent_type alphazero --amp --num_workers 8 --iterations 500 --games_per_iteration 500 --eval_games 200 --learning_rate 2e-4 --num_simulations 200 --value_coef 2.5 --c_puct 2.0 --self_play_temp 1.0 --temp_moves 30 --league_prob 0.2 --heuristic_prob 0.7 --az_best_by heuristic --az_eval_freq 10 --az_eval_games_best 200 --root_dirichlet_alpha 0.3 --root_dirichlet_epsilon 0.25
```

```bash
uv run python train.py --board_size 9 --agent_type alphazero --amp --num_workers 1 --worker_device cuda --no-compile --mcts_batch_size 64 --batch_size 128 --iterations 300 --games_per_iteration 200 --eval_games 200 --learning_rate 2e-4 --num_simulations 200 --value_coef 2.5 --c_puct 2.0 --self_play_temp 1.0 --temp_moves 30 --league_prob 0.2 --heuristic_prob 0.4 --az_best_by heuristic --az_eval_freq 10 --az_eval_games_best 200 --root_dirichlet_alpha 0.3 --root_dirichlet_epsilon 0.25 --resume weights/checkpoint_635.pt
```

### DQN 9x9 initial training

```bash
uv run python train.py --board_size 9 --agent_type dqn --amp --num_workers 8 --iterations 2000 --games_per_iteration 1000 --learning_rate 2e-4 --gamma 0.99 --replay_buffer_size 100000 --league_prob 0.25 --heuristic_prob 0.2
```

### DQN after 100% random success

```bash
uv run python train.py --board_size 9 --agent_type dqn --amp --num_workers 8 --iterations 2000 --games_per_iteration 1000 --learning_rate 2e-4 --gamma 0.99 --replay_buffer_size 100000 --league_prob 0.2 --heuristic_prob 0.6 --resume weights/dqn_checkpoint_1030.pt
```

### DQN heuristic-focused fine-tune

```bash
uv run python train.py --board_size 9 --agent_type dqn --best_by heuristic --amp --num_workers 8 --iterations 2000 --games_per_iteration 1000 --eval_games 200 --learning_rate 2e-4 --gamma 0.99 --replay_buffer_size 100000 --league_prob 0.2 --heuristic_prob 0.3 --heuristic_win_bonus 0.2 --heuristic_prob_start 1 --heuristic_prob_decay_iters 500 --resume weights/dqn_checkpoint_290.pt
```

### DQN training tips

- **Longer runs**: Use 500+ iterations and a large replay buffer (e.g. `--replay_buffer_size 200000`) so value can propagate from terminal (+1/-1) back through many steps.
- **Metrics beyond loss**: Watch the eval line (`Eval: vs_random win_rate=... vs_heuristic win_rate=...`). Low loss with low win rate means the network is fitting targets that do not yet encode a good policy; loss alone can be misleading with sparse rewards.
- **Terminal oversampling**: Default `--dqn_terminal_fraction 0.5` increases the fraction of batches drawn from game-ending transitions so the TD target carries real value.
- **Best by heuristic**: Use `--best_by heuristic` (default) so the saved best model is the one with highest win rate vs the heuristic bot, not lowest loss. Use `--eval_games_best 100` (or 200) for a more stable choice.
- **Beating the heuristic**: If vs_heuristic win rate stays low, try `--heuristic_win_bonus 0.2` to give extra reward for wins vs heuristic, or a curriculum: `--heuristic_prob_start 0.8 --heuristic_prob_decay_iters 500` to start with more heuristic games then decay to `--heuristic_prob`.
- **Loss stuck near 0**: If loss is ~0 but win rate is poor, try a slightly higher `--learning_rate` (e.g. 3e-4 or 5e-4) or longer `--epsilon_decay_steps` so the policy keeps exploring.
