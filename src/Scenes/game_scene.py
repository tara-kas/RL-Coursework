import pygame
import math
import numpy as np
import random

from src.Scenes.scene import Scene
from src.scene_manager import SceneManager
from src.game_logic import GameLogic

COLOURS = {
    "bg_dark": (28, 26, 31),
    "frame_dark": (61, 41, 20),
    "frame_wood": (82, 55, 28),
    "frame_light": (120, 82, 42),
    "board_surface": (232, 220, 196),
    "grid_line": (139, 115, 85),
    "stone_outline": (30, 28, 26),
    "stone_highlight": (255, 255, 255),
    "ui_panel": (45, 40, 38),
    "ui_text": (240, 235, 230),
    "ui_turn_glow": (255, 212, 80),
}


class GameScene(Scene):
    def __init__(self, scene_manager: SceneManager):
        super().__init__(scene_manager)
        
        self.padding_x_L = 24
        self.padding_x_R = 24
        self.padding_y_T = 72
        self.padding_y_B = 24
        
        screen_width, screen_height = scene_manager.screen.get_size()
        
        self.panel_top_x = self.padding_x_L
        self.panel_top_y = self.padding_y_T
        self.panel_bottom_x = screen_width - self.padding_x_R
        self.panel_bottom_y = screen_height - self.padding_y_B
        
        self.grid_x = 15
        self.grid_y = 15
        
        # 15x15 playable grid in the middle
        num_gaps = self.grid_x - 1  # 14 gaps between 15 intersections
        num_slots = num_gaps + 2   # 16 slots for grid and margin
        
        self.board_padding = 14
        
        avail_w = (self.panel_bottom_x - self.panel_top_x) - 2 * self.board_padding
        avail_h = (self.panel_bottom_y - self.panel_top_y) - 2 * self.board_padding
        
        tile_size = min(avail_w / num_slots, avail_h / num_slots)
        
        self.tile_size = tile_size
        self.pixel_scale_factor = tile_size / 16
        
        # Board surface is 16x16 tiles to include grid and margin
        play_w = num_slots * tile_size
        play_h = num_slots * tile_size
        
        self.play_top_x = self.panel_top_x + ((self.panel_bottom_x - self.panel_top_x) - play_w - 2 * self.board_padding) / 2 + self.board_padding
        self.play_top_y = self.panel_top_y + ((self.panel_bottom_y - self.panel_top_y) - play_h - 2 * self.board_padding) / 2 + self.board_padding
        self.play_bottom_x = self.play_top_x + play_w
        self.play_bottom_y = self.play_top_y + play_h
        
        self.frame_rect = pygame.Rect(
            self.play_top_x - self.board_padding,
            self.play_top_y - self.board_padding,
            play_w + 2 * self.board_padding,
            play_h + 2 * self.board_padding,
        )
        
        # Playable 15x15 grid is inset by one tile from the board surface (not on the outline)
        self.top_x = self.play_top_x + tile_size
        self.top_y = self.play_top_y + tile_size
        self.bottom_x = self.top_x + num_gaps * tile_size
        self.bottom_y = self.top_y + num_gaps * tile_size

        users = [{"type": "player", "name": "player1", "colour": (0,0,255)}, {"type": "bot", "name": "Mr Random", "file": "random", "colour": (255,0,0)}]
        self.game_logic = GameLogic(users=users)
        
        self.mouse_x = 0
        self.mouse_y = 0

        self.font_size = max(14, int(13 * self.pixel_scale_factor))
        self.font = pygame.font.Font("assets/fonts/Pixelify_Sans/static/PixelifySans-Bold.ttf", self.font_size)
        self.font_small = pygame.font.Font("assets/fonts/Pixelify_Sans/static/PixelifySans-Regular.ttf", max(11, self.font_size - 2))

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
                    # Snap to nearest intersection of grid lines
                    grid_x = int(round((self.mouse_x - self.top_x) / self.tile_size))
                    grid_y = int(round((self.mouse_y - self.top_y) / self.tile_size))
                    
                    if grid_x < 0 or grid_y < 0 or grid_x >= self.grid_x or grid_y >= self.grid_y:
                        continue
                    
                    if self.game_logic.check_valid_move((grid_x, grid_y)):
                        self.game_logic.make_move(self.game_logic.current_turn, (grid_x, grid_y))
                        self.game_logic.next_turn()

    def update(self):
        pass

    def draw(self):
        self.draw_background()
        self.draw_stones()
        self.draw_ui()

    def draw_background(self):
        screen = self.scene_manager.screen
        screen.fill(COLOURS["bg_dark"])

        self.draw_board_frame()

        board_rect = pygame.Rect(self.play_top_x, self.play_top_y, self.play_bottom_x - self.play_top_x, self.play_bottom_y - self.play_top_y)
        pygame.draw.rect(screen, COLOURS["board_surface"], board_rect)

        grid_colour = COLOURS["grid_line"]
        line_w = max(1, int(round(self.tile_size / 24)))
        
        # 15 vertical lines
        for i in range(self.grid_x):
            x = int(self.top_x + i * self.tile_size)
            pygame.draw.line(screen, grid_colour, (x, self.top_y), (x, self.bottom_y), line_w)
            
        # 15 horizontal lines
        for j in range(self.grid_y):
            y = int(self.top_y + j * self.tile_size)
            pygame.draw.line(screen, grid_colour, (self.top_x, y), (self.bottom_x, y), line_w)
            
    def draw_board_frame(self):
        screen = self.scene_manager.screen
        r = self.frame_rect
        
        # Board frame shadow
        shadow = r.inflate(4, 4)
        shadow.x += 3
        shadow.y += 3
        pygame.draw.rect(screen, (18, 16, 14), shadow, border_radius=0)
        
        # Board frame dark edge
        pygame.draw.rect(screen, COLOURS["frame_dark"], r, border_radius=0)
        
        # Board frame bevel inner highlight
        inner = r.inflate(-4, -4)
        pygame.draw.line(screen, COLOURS["frame_light"], (r.x + 4, r.y + 4), (r.x + r.w - 5, r.y + 4), 2)
        pygame.draw.line(screen, COLOURS["frame_light"], (r.x + 4, r.y + 4), (r.x + 4, r.y + r.h - 5), 2)
        
        # Board frame inner shadow
        pygame.draw.line(screen, COLOURS["frame_dark"], (r.x + 4, r.y + r.h - 5), (r.x + r.w - 5, r.y + r.h - 5), 2)
        pygame.draw.line(screen, COLOURS["frame_dark"], (r.x + r.w - 5, r.y + 4), (r.x + r.w - 5, r.y + r.h - 5), 2)
        
        # Board frame wood fill
        pygame.draw.rect(screen, COLOURS["frame_wood"], inner, border_radius=0)

    def draw_stones(self):
        screen = self.scene_manager.screen
        
        outline = COLOURS["stone_outline"]
        radius = max(2, (self.tile_size - 6) // 2)
        outline_w = max(1, int(radius * 0.12))
        
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                cell_value = self.game_logic.game_state.board[i, j]
                
                if cell_value == -1:
                    continue
                
                # Stone on intersection of grid lines
                cx = int(self.top_x + i * self.tile_size) + 1
                cy = int(self.top_y + j * self.tile_size) + 1
                
                colour = self.game_logic.users[cell_value]["colour"]
                
                pygame.draw.circle(screen, outline, (cx, cy), radius + outline_w)
                pygame.draw.circle(screen, colour, (cx, cy), radius)

    def draw_ui(self):
        screen = self.scene_manager.screen
        header_h = self.padding_y_T - 8
        header_rect = pygame.Rect(0, 0, screen.get_width(), header_h)
        
        # Player Panel
        pygame.draw.rect(screen, COLOURS["ui_panel"], header_rect)
        pygame.draw.line(screen, COLOURS["frame_light"], (0, 0), (screen.get_width(), 0), 2)
        pygame.draw.line(screen, (60, 55, 52), (0, header_h - 1), (screen.get_width(), header_h - 1), 1)
        
        # Player badges
        badge_h = 40
        badge_y = (header_h - badge_h) // 2
        start_x = self.padding_x_L + 8
        gap = 12
        
        for index, user in enumerate(self.game_logic.users):
            is_current = index == self.game_logic.current_turn
            name = user["name"]
            type = user["type"]
            colour = user["colour"]
            badge_w = max(100, self.font.size(name)[0] + self.font_small.size(f" ({type})")[0] + 24)
            badge_rect = pygame.Rect(start_x, badge_y, badge_w, badge_h)
            
            if is_current:
                pygame.draw.rect(screen, (60, 55, 52), badge_rect.inflate(4, 4))
                pygame.draw.rect(screen, COLOURS["ui_turn_glow"], badge_rect.inflate(4, 4), 2)
                
            pygame.draw.rect(screen, (55, 50, 48), badge_rect)
            pygame.draw.rect(screen, (70, 65, 62), badge_rect, 1)
            
            # Stone colour icon
            stone_r = 6
            stone_x = badge_rect.x + 14
            stone_y = badge_rect.centery
            pygame.draw.circle(screen, COLOURS["stone_outline"], (stone_x, stone_y), stone_r + 1)
            pygame.draw.circle(screen, colour, (stone_x, stone_y), stone_r)
            
            # Player name and type
            text_colour = COLOURS["ui_turn_glow"] if is_current else COLOURS["ui_text"]
            text_surf = self.font_small.render(f"{name} ({type})", True, text_colour)
            screen.blit(text_surf, (badge_rect.x + 28, badge_rect.centery - text_surf.get_height() // 2))
            start_x += badge_w + gap
            
        if self.game_logic.users[self.game_logic.current_turn]["type"] == "player":
            self.draw_hover_effects(self.mouse_x, self.mouse_y)

    def draw_hover_effects(self, mouse_x: int, mouse_y: int):
        # Snap to nearest intersection of grid lines
        grid_x = int(round((mouse_x - self.top_x) / self.tile_size))
        grid_y = int(round((mouse_y - self.top_y) / self.tile_size))
        
        if grid_x < 0 or grid_y < 0 or grid_x >= self.grid_x or grid_y >= self.grid_y:
            return
        
        cx = int(self.top_x + grid_x * self.tile_size) + 1
        cy = int(self.top_y + grid_y * self.tile_size) + 1
        
        radius = max(4, (self.tile_size - 6) // 2)
        
        # Hover ring
        hover_surf = pygame.Surface((radius * 2 + 8, radius * 2 + 8))
        hover_surf.set_colorkey((0, 0, 0))
        
        pygame.draw.circle(hover_surf, (255, 255, 255), (radius + 4, radius + 4), radius + 2)
        hover_surf.set_alpha(72)
        
        self.scene_manager.screen.blit(hover_surf, (cx - radius - 4, cy - radius - 4))
        pygame.draw.circle(self.scene_manager.screen, (255, 255, 255), (cx, cy), radius + 2, 2)