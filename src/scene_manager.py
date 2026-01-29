import pygame

class SceneManager():
    def __init__(self, screen:pygame.Surface, change_scene_callback, shutdown_callback):
        """
        Args:
            screen: The pygame screen surface for drawing
            change_scene_callback: Function to call when changing scenes
            shutdown_callback: Function to call when shutting down
        """
        self.screen = screen
        self._change_scene_callback = change_scene_callback
        self._shutdown_callback = shutdown_callback
    
    def change_scene(self, scene_name:str):
        """
        Change to a different scene.
        """
        self._change_scene_callback(scene_name)

    def shutdown(self):
        """
        Shutdown the application.
        """
        self._shutdown_callback()
