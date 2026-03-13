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
    parser.add_argument(
        "--board_size",
        type=int,
        default=15,
        choices=[9, 15],
        help="Board size (must be 9 or 15)",
    )
    args = parser.parse_args()

    window = Window(720, 720, "Game", "game", args.weights, args.board_size)

    while window.running:
        window.handle_events()
        window.update()
        window.draw()
        pygame.time.Clock().tick(FPS)


if __name__ == "__main__":
    main()