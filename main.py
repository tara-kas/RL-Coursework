import argparse
import pygame
from src.window import Window

FPS = 60

def main():
    parser = argparse.ArgumentParser(description="Gomoku UI")
    parser.add_argument("--board_size", type=int, default=15, choices=[9, 15])
    parser.add_argument(
        "--bot_file",
        type=str,
        default="ppo_gomoku_model_15x15",
        help="Bot module file in src/Bots without .py (e.g. ppo_gomoku_model_15x15, random)",
    )
    parser.add_argument("--bot_name", type=str, default="PPO Bot")
    parser.add_argument(
        "--weights_path",
        type=str,
        default="p0_ppo.pt",
        help="Optional checkpoint path passed to bot as weights_path",
    )
    parser.add_argument(
        "--bot_kwargs",
        type=str,
        default="{}",
        help='Extra JSON kwargs for bot constructor, e.g. \'{"num_simulations":500}\'',
    )
    args = parser.parse_args()

    import json

    try:
        bot_kwargs = json.loads(args.bot_kwargs)
        if not isinstance(bot_kwargs, dict):
            raise ValueError("bot_kwargs must decode to a JSON object")
    except Exception as exc:
        raise ValueError(f"Invalid --bot_kwargs JSON: {exc}") from exc

    window = Window(
        width=720,
        height=720,
        current_scene="game",
        board_size=args.board_size,
        bot_file=args.bot_file,
        bot_name=args.bot_name,
        weights_path=args.weights_path,
        bot_kwargs=bot_kwargs,
    )

    while window.running:
        window.handle_events()
        window.update()
        window.draw()
        pygame.time.Clock().tick(FPS)


if __name__ == "__main__":
    main()
