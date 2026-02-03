import pygame
import math
import numpy as np
import random

from src.Scenes.scene import Scene
from src.scene_manager import SceneManager


class GameScene(Scene):
    def __init__(self, scene_manager: SceneManager):
        super().__init__(scene_manager)
        
        self.padding_x_L = 20
        self.padding_x_R = 20
        self.padding_y_T = 60
        self.padding_y_B = 20
        
        screen_width, screen_height = scene_manager.screen.get_size()
        
        self.panel_top_x = self.padding_x_L
        self.panel_top_y = self.padding_y_T
        self.panel_bottom_x = screen_width - self.padding_x_R
        self.panel_bottom_y = screen_height - self.padding_y_B
        
        self.grid_x = 15
        self.grid_y = 15
        
        tile_size = min((self.panel_bottom_x - self.panel_top_x) / self.grid_x, (self.panel_bottom_y - self.panel_top_y) / self.grid_y)
        self.tile_size = tile_size
        self.pixel_scale_factor = tile_size / 16
        
        self.top_x = self.panel_top_x
        self.top_y = self.panel_top_y
        self.bottom_x = self.top_x + (self.grid_x * tile_size)
        self.bottom_y = self.top_y + (self.grid_y * tile_size)
        
        background_tile = pygame.image.load("assets/textures/background.png")
        self.background_tile = pygame.transform.scale(background_tile, (math.ceil(tile_size), math.ceil(tile_size)))

        self.users = [{"type": "player", "name": "Test Player"}, {"type": "bot", "name": "Test Bot"}]

        self.current_turn = random.randint(0, len(self.users) - 1)

    def handle_events(self, events: list[pygame.event.Event]):
        pass

    def update(self):
        pass

    def draw(self):
        self.draw_background()
        self.draw_entities()
        self.draw_ui()

    def draw_background(self):
        #Draw tiled background
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                x = self.top_x + i * self.tile_size
                y = self.top_y + j * self.tile_size
                self.scene_manager.screen.blit(self.background_tile, (x, y))
                
        #Vertical grid lines
        for i in range(self.grid_x + 1):
            x = self.top_x + i * self.tile_size
            pygame.draw.line(self.scene_manager.screen, (50, 50, 50), (x, self.top_y), (x, self.bottom_y), 1)
            
        #Horizontal grid lines
        for j in range(self.grid_y + 1):
            y = self.top_y + j * self.tile_size
            pygame.draw.line(self.scene_manager.screen, (50, 50, 50), (self.top_x, y), (self.bottom_x, y), 1)

    def draw_entities(self):
        pass

    def draw_ui(self):
        font_size = int(12 * self.pixel_scale_factor)
        font = pygame.font.SysFont("Arial", font_size)

        line_length = 0
        
        for index, user in enumerate(self.users):
            if index == self.current_turn:
                text_color = (255, 215, 0)  # Gold color for current turn
            else:
                text_color = (255, 255, 255)  # White color for others

            user_text = f"{user['name']} ({user['type']})"

            text_surface = font.render(user_text, True, text_color)

            x = self.padding_x_L + line_length
            y = self.padding_y_T / 2 - font_size / 2

            self.scene_manager.screen.blit(text_surface, (x, y))

            line_length += len(user["name"]) * font_size