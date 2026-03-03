import pygame
import math
import numpy as np
import random

from src.Scenes.scene import Scene
from src.scene_manager import SceneManager
from src.game_logic import GameLogic


class GameScene(Scene):
    def __init__(self, scene_manager: SceneManager):
        super().__init__(scene_manager)

        users = [{"type": "player", "name": "player1", "colour": (0,0,255)}, {"type": "bot", "name": "AlphaZero", "file": "alpha_zero_transform", "colour": (255,0,0)}]
        self.game_logic = GameLogic(users=users)

    def handle_events(self, events: list[pygame.event.Event]):
        if self.game_logic.users[self.game_logic.current_turn]["type"] == "bot":
            bot_name = self.game_logic.users[self.game_logic.current_turn]["name"]
            move = self.game_logic.get_bot_move(bot_name)

            if self.game_logic.check_valid_move(move):
                self.game_logic.make_move(self.game_logic.current_turn, move)
                self.game_logic.next_turn()
        
        for event in events:
            if hasattr(event, "pos"):
                self.mouse_x, self.mouse_y = event.pos
            
            if self.game_logic.users[self.game_logic.current_turn]["type"] == "player":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    grid_x = int((self.mouse_x - self.top_x) // self.tile_size)
                    grid_y = int((self.mouse_y - self.top_y) // self.tile_size)
                    
                    if grid_x < 0 or grid_y < 0 or grid_x >= self.grid_x or grid_y >= self.grid_y:
                        continue
                    
                    if self.game_logic.check_valid_move((grid_x, grid_y)):
                        self.game_logic.make_move(self.game_logic.current_turn, (grid_x, grid_y))
                        self.game_logic.next_turn()

    def update(self):
        pass

    def draw(self):
        pass