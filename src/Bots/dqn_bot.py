import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState
from src.gomoku_game import get_legal_moves

import random

import copy
import keras
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras import optimizers

class DQNBot(BaseBot):
    def __init__(self, board_size: int = 15, action_size: int = 225):
        self.board_size = board_size
        self.action_size = action_size
        
        # initialise replay buffer that stores (action, state, reward, next_state, done) w/
        # max 8000 past games
        self.memory = deque(maxlen=8000)
        
        # discount factor
        self.gamma = 0.75
        
        # exploration/exploitation
        self.epsilon = 1e-07
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.99995
        
        self.learning_rate = 0.00001
        self.model = self._build_model()
        
    def _build_model(self):
        # neural network for deep-Q
        model = Sequential()
        
        # input layer
        model.add(Dense(1024, input_dim=self.state_size, activation="linear"))
        model.add(LeakyReLU(negative_slope=0.01))
        model.add(BatchNormalization())

        # hidden layers
        model.add(Dense(1024, activation="linear"))
        model.add(LeakyReLU(negative_slope=0.01))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(512, activation="linear"))
        model.add(LeakyReLU(negative_slope=0.01))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(512, activation="linear"))
        model.add(LeakyReLU(negative_slope=0.01))
        model.add(BatchNormalization())

        model.add(Dense(256, activation="linear"))
        model.add(LeakyReLU(negative_slope=0.01))
        model.add(BatchNormalization())

        model.add(Dense(256, activation="linear"))
        model.add(LeakyReLU(negative_slope=0.01))
        
        # output layer
        model.add(Dense(self.action_size, activation="linear"))

        model.compile(loss='mse',optimizer=keras.optimizers.RMSprop(lr=self.learning_rate,rho=0.9, epsilon=self.epsilon, decay=self.epsilon_decay))
    
        return model    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        
