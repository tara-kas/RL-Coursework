import pygame
import math
import numpy as np

from src.Scenes.scene import Scene
from src.scene_manager import SceneManager


class GameScene(Scene):
    def __init__(self, scene_manager: SceneManager):
        super().__init__(scene_manager)

    def handle_events(self, events: list[pygame.event.Event]):
        pass

    def update(self):
        pass

    def draw(self):
        self.draw_background()
        self.draw_entities()

    def draw_background(self):
        pass

    def draw_entities(self):
        pass