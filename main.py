import pygame
from src.window import Window

FPS = 60

def main():
    window = Window(width=720, height=720)

    while window.running:
        window.handle_events()
        window.update()
        window.draw()
        pygame.time.Clock().tick(FPS)


if __name__ == "__main__":
    main()