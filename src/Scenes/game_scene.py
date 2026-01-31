import pygame
import math
import numpy as np

from src.Scenes.scene import Scene
from src.scene_manager import SceneManager


class GameScene(Scene):
    def __init__(self, scene_manager: SceneManager):
        super().__init__(scene_manager)
        
        self.padding_x_L = 20
        self.padding_x_R = 20
        self.padding_y_T = 20
        self.padding_y_B = 20
        
        screen_width, screen_height = scene_manager.screen.get_size()
        
        self.top_x = self.padding_x_L
        self.top_y = self.padding_y_T
        self.bottom_x = screen_width - self.padding_x_R
        self.bottom_y = screen_height - self.padding_y_B
        
        self.grid_x = 15
        self.grid_y = 15
        
        tile_size = min((self.bottom_x - self.top_x) / self.grid_x, (self.bottom_y - self.top_y) / self.grid_y)
        self.tile_size = tile_size
        self.pixel_scale_factor = tile_size / 16
        
        background_tile = pygame.image.load("assets/textures/background.png")
        self.background_tile = pygame.transform.scale(background_tile, (math.ceil(tile_size), math.ceil(tile_size)))

    def handle_events(self, events: list[pygame.event.Event]):
        pass

    def update(self):
        pass

    def draw(self):
        self.draw_background()
        self.draw_entities()

    def draw_background(self):
        #Draw tiled background
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                x = self.top_x + i * self.tile_size
                y = self.top_y + j * self.tile_size
                self.scene_manager.screen.blit(self.background_tile, (x, y))
                        

    def draw_entities(self):
        pass