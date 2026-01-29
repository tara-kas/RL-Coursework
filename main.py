import pygame
from src.window import Window

FPS = 60

def main():
    window = Window()

    while window.running:
        window.handle_events()
        window.update()
        window.draw()
        pygame.time.Clock().tick(FPS)


if __name__ == "__main__":
    main()