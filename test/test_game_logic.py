from src.game_logic import GameLogic
import numpy as np

users = [{"type": "player", "name": "p1", "colour": (0,0,255)}, {"type": "player", "name": "p2", "colour": (0,255,0)}]
n = 15
m = 15
mock = GameLogic(n,m,users)

n_1 = 4
m_1 = 4
mock_valids = GameLogic(n_1,m_1,users)
mock_valids.game_state = np.array([[-1,0,-1,0],
                                   [0,0,0,0],
                                   [0,0,0,0],
                                   [0,0,0,0]])
game_state_2 = np.array([[-1,-1,-1,-1],
                         [0,0,-1,0],
                         [0,0,0,0],
                         [-1,0,0,0]])

def test_check_valid_move():
    assert mock.check_valid_move((-1,0)) == False
    assert mock.check_valid_move((-1,-10)) == False
    assert mock.check_valid_move((0,0)) == True
    assert mock.check_valid_move((0,m-1)) == True
    assert mock.check_valid_move((n-1,0)) == True
    assert mock.check_valid_move((n-1,m-1)) == True
    assert mock.check_valid_move((n,m-1)) == False

def test_make_move():
    pass

def test_get_valid_moves():
    assert mock_valids.get_valid_moves() == [(0,0), (0,2)]
    mock_valids.game_state = game_state_2
    assert mock_valids.get_valid_moves() == [(0,0), (0,1), (0,2), (0,3), (1,2), (3,0)]