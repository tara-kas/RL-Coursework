import argparse
import pygame
from src.window import Window

FPS = 60


def main():
    parser = argparse.ArgumentParser(description="Gomoku game (player vs AlphaZero)")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to checkpoint. Use weights/best.pt for AlphaZero or weights/dqn_best.pt for DQN. Auto-detects model from path.",
    )
    args = parser.parse_args()

    window = Window(width=720, height=720, current_scene="game", weights_path=args.weights)

    while window.running:
        window.handle_events()
        window.update()
        window.draw()
        pygame.time.Clock().tick(FPS)


if __name__ == "__main__":
    main()