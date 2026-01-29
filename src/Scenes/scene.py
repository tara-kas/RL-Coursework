import pygame
from abc import ABC, abstractmethod

from src.scene_manager import SceneManager

class Scene(ABC):
    def __init__(self, scene_manager: SceneManager):
        self.scene_manager = scene_manager

    @abstractmethod
    def handle_events(self, events:list[pygame.event.Event]):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def draw(self):
        pass