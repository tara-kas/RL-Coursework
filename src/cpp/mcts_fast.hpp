#ifndef MCTS_FAST_HPP
#define MCTS_FAST_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <utility>
#include <vector>
#include <cmath>

#include "game_utils.hpp"
namespace py = pybind11;

struct MCTSNode {
    int parentIdx = -1;
    int firstChildIdx = -1;
    short numChildren = 0;

    short lastMove = -1; //0 to 255 translated
    short curPlayer = -1;

    int N = 0;
    float W = 0;
    float P = 0;
};

class MCTS {
private:
    std::vector<MCTSNode> arena;
    int rootIdx;

    std::vector<std::vector<int>> batchPaths;
    std::vector<int> batchLeafIdxs;

    std::vector<int> rootBoard;
    int rootPlayer;

public:
    MCTS() {
        arena.reserve(5000000);
        rootIdx = allocateNode(-1, -1, 0, 0.0);
    };

    void setRootState(py::array_t<float> pyBoard, int curPlayer) {
        auto fastBoard = pyBoard.unchecked<2>();
        
        rootBoard.resize(BOARD_AREA); // 9x9 = 81 flat array
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                rootBoard[r * BOARD_SIZE + c] = fastBoard(r, c);
            }
        }
        rootPlayer = curPlayer;
        
        // Reset the tree
        arena.clear();
        rootIdx = allocateNode(-1, -1, curPlayer, 1.0f);
    }

    int allocateNode(int parent, short move, short player, float prior) {
        arena.push_back(MCTSNode());
        int newIdx = arena.size() - 1;
        arena[newIdx].parentIdx = parent;
        arena[newIdx].lastMove = move;
        arena[newIdx].curPlayer = player;
        arena[newIdx].P = prior;

        return newIdx;
    }

    MCTSNode& getNode(int idx) {
        return arena[idx];
    }

    void reset() {
        arena.clear();
        rootIdx = allocateNode(-1, -1, 0, 0.0);
    }

    std::tuple<int, std::vector<int>, int> selectLeaf(float cPuct, std::vector<int>& simBoard, int& simPlayer) {
        int curIdx = rootIdx;
        std::vector<int> path = {};

        while(arena[curIdx].numChildren > 0) {
            path.push_back(curIdx);
            arena[curIdx].N += 1;
            arena[curIdx].W -= 1.0f;

            int parentVisits = arena[curIdx].N - 1;
            
            float bestScore = -std::numeric_limits<float>::infinity();
            int bestChildIdx = -1;
            
            for(int i = 0; i < arena[curIdx].numChildren; ++i) {
                int childIdx = arena[curIdx].firstChildIdx + i;
                
                float q = 0.0f;
                if (arena[childIdx].N > 0) {
                    q = arena[childIdx].W / static_cast<float>(arena[childIdx].N);
                }
                
                float p = arena[childIdx].P;
                float u = cPuct * p * std::sqrt(static_cast<float>(parentVisits)) / (1.0f + arena[childIdx].N);
                
                float score = q + u;
                
                if (score > bestScore) {
                    bestScore = score;
                    bestChildIdx = childIdx;
                }
            }

            if(bestChildIdx == -1) {
                break;
            }

            curIdx = bestChildIdx;
            
            int move = arena[curIdx].lastMove;
            simBoard[move] = simPlayer;
            simPlayer = 1 - simPlayer;
        }

        path.push_back(curIdx);
        arena[curIdx].N += 1;
        arena[curIdx].W -= 1.0f;

        int termVal = getGameResult(simBoard, arena[curIdx].lastMove, arena[curIdx].curPlayer);
        
        return {curIdx, path, termVal};
    }

    void backupPath(const std::vector<int>& path, float value) {
        for(int i = path.size() - 1; i >=0 ; --i) {
            arena[path[i]].W += (value + 1.0f);
            value = - value;
        }
    }

    py::array_t<float> gatherBatch(int batchSize, float cPuct) {
        batchPaths.clear();
        batchLeafIdxs.clear();

        py::array_t<float> batchedBoards({batchSize, BOARD_SIZE, BOARD_SIZE});
        auto mutableBoards = batchedBoards.mutable_unchecked<3>();

        int validLeaves = 0;

        for(int i = 0; i < batchSize; ++i) {
            std::vector<int> simBoard = rootBoard;
            int simPlayer = rootPlayer;

            auto [leafIdx, path, termVal] = selectLeaf(cPuct, simBoard, simPlayer);

            if(termVal != -1) {
                backupPath(path, termVal);
                continue;
            }

            batchPaths.push_back(path);
            batchLeafIdxs.push_back(leafIdx);

            for(int x = 0; x < BOARD_SIZE; ++x) {
                for(int y = 0; y < BOARD_SIZE; ++y) {
                    mutableBoards(validLeaves, x, y) = static_cast<float>(simBoard[x*BOARD_SIZE + y]);
                }
            }
            validLeaves++;
        }

        batchedBoards.resize({validLeaves, BOARD_SIZE, BOARD_SIZE});
        return batchedBoards;

    }

    void expandBackup(py::array_t<float> policiesPy, py::array_t<float> valuesPy) {
        auto policies = policiesPy.unchecked<2>();
        auto values = valuesPy.unchecked<1>();

        for(int i = 0; i < batchLeafIdxs.size(); ++i) {
            int leafIdx = batchLeafIdxs[i];
            float value = values(i);

            arena[leafIdx].firstChildIdx = arena.size();

            short numLegal = 0;

            for(short move = 0; move < BOARD_AREA; ++move) {
                float p = policies(i, move);
                if(p > 1e-6) {
                    allocateNode(leafIdx, move, 1 - arena[leafIdx].curPlayer, p);
                    numLegal++;
                }
            }
            arena[leafIdx].numChildren = numLegal;
            backupPath(batchPaths[i], value);
        }
    }

    std::vector<int> getRootVisits() {
        std::vector<int> visits(BOARD_AREA, 0); 

        for (int i = 0; i < arena[rootIdx].numChildren; ++i) {
            int childIdx = arena[rootIdx].firstChildIdx + i;
            int move = arena[childIdx].lastMove;
            visits[move] = arena[childIdx].N;
        }
        return visits;
    }

    float getRootValue() {
        if (arena.empty() || rootIdx == -1 || arena[rootIdx].N == 0) {
            return 0.0f;
        }
        return arena[rootIdx].W / static_cast<float>(arena[rootIdx].N);
    }

    void changeRoot(short lastMove, py::array_t<float> pyBoard, int curPlayer) {
        int newRootIdx = -1;
        
        if (rootIdx != -1 && arena.size() > 0) {
            for (int i = 0; i < arena[rootIdx].numChildren; ++i) {
                int childIdx = arena[rootIdx].firstChildIdx + i;
                if (arena[childIdx].lastMove == lastMove) {
                    newRootIdx = childIdx;
                    break;
                }
            }
        }

        if (newRootIdx != -1) {
            rootIdx = newRootIdx;
            arena[rootIdx].parentIdx = -1; // Sever the parent link so backupPath doesn't go too high
            
            auto fastBoard = pyBoard.unchecked<2>();
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    rootBoard[r * BOARD_SIZE + c] = fastBoard(r, c);
                }
            }
            rootPlayer = curPlayer;
        } else {
            setRootState(pyBoard, curPlayer);
        }
    }

    ~MCTS() = default;
};



#endif