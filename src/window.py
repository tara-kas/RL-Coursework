import pygame
import sys
from src.scene_manager import SceneManager

class Window():
    def __init__(self, width:int=1280, height:int=720, title:str="Game", current_scene:str="game"):
        pygame.init()
        self.width = width
        self.height = height
        self.title = title
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self.running = True
        self.current_scene = current_scene
        
        self.scene_manager = SceneManager(
            self.screen,
            self.change_scene,
            self.shutdown
        )
        
        from src.Scenes.game_scene import GameScene

        self.scenes = {
            "game": GameScene(self.scene_manager)
        }

    def handle_events(self):
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT:
                self.shutdown()
  
        self.scenes[self.current_scene].handle_events(events)

    def update(self):
        self.scenes[self.current_scene].update()
        pygame.display.update()

    def draw(self):
        # Clear the screen with black background
        self.screen.fill((0, 0, 0))

        # Draw the current scene
        self.scenes[self.current_scene].draw()
    
    def change_scene(self, scene_name: str):
        if scene_name in self.scenes:
            self.current_scene = scene_name

    def shutdown(self):
        self.running = False
        pygame.quit()
        sys.exit()

    