#ifndef GAME_HELPERS_HPP
#define GAME_HELPERS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <utility>
#include <stdexcept>

#define BOARD_SIZE 9
#define BOARD_AREA (BOARD_SIZE * BOARD_SIZE)

namespace py = pybind11;

int getGameResult(std::vector<int>& fastBoard, short lastMove, int lastPlayer) {
    if (lastMove < 0 || lastMove >= BOARD_AREA) return -1;
    
    int newX = lastMove / BOARD_SIZE;
    int newY = lastMove % BOARD_SIZE;

    int count = 0;
    for(int cell = 0; cell < 9; ++cell) {
        int nx = newX - 4 + cell;
        if(nx >= 0 && nx < BOARD_SIZE) {
            if(fastBoard[nx * BOARD_SIZE + newY] == lastPlayer) {
                count++;
                if(count == 5) {
                    return 1;
                }
                else {
                    count = 0;
                }
            }
        }
    }

    count = 0;
    for(int cell = 0; cell < 9; ++cell) {
        int ny = newY - 4 + cell;
        if(ny >= 0 && ny < BOARD_SIZE) {
            if(fastBoard[newX * BOARD_SIZE + ny] == lastPlayer) {
                count++;
                if(count == 5) {
                    return 1;
                }
                else {
                    count = 0;
                }
            }
        }
    }
    
    count = 0;
    for(int cell = 0; cell < 9; ++cell) {
        int nx = newX - 4 + cell;
        int ny = newY - 4 + cell;
        if(nx >= 0 && nx < BOARD_SIZE && ny >=0 && ny < BOARD_SIZE) {
            if(fastBoard[nx * BOARD_SIZE + ny] == lastPlayer) {
                count++;
                if(count == 5) {
                    return 1;
                }
                else {
                    count = 0;
                }
            }
        }
    }
    
    count = 0;
    for(int cell = 0; cell < 9; ++cell) {
        int nx = newX + 4 - cell;
        int ny = newY - 4 + cell;
        if(nx >= 0 && nx < BOARD_SIZE && ny >=0 && ny < BOARD_SIZE) {
            if(fastBoard[nx * BOARD_SIZE + ny] == lastPlayer) {
                count++;
                if(count == 5) {
                    return 1;
                }
                else {
                    count = 0;
                }
            }
        }
    }

    bool isFull = true;
    for (int i = 0; i < BOARD_AREA; ++i) {
        if (fastBoard[i] == -1) {
            isFull = false;
            break;
        }
    }
    if (isFull) {
        return 0; // DRAW
    }

    return -1;
}

std::vector<std::pair<int, int>> getLegalMoves(py::array_t<float> board) {
    py::buffer_info buf = board.request();

    if(buf.ndim != 2) {
        throw std::runtime_error("Wrong dimensions of board");
    }

    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];

    auto fastBoard = board.unchecked<2>();
    std::vector<std::pair<int, int>> legalMoves;
    legalMoves.reserve(rows * cols);

    for(int x = 0; x < rows; ++x) {
        for(int y = 0; y < cols; ++y) {
            if(fastBoard(x, y) == -1) {
                legalMoves.push_back({x, y});
            }
        }
    }

    return legalMoves;
}

#endif