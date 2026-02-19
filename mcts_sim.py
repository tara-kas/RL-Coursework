from src.Bots.mcts import Bot

ts = Bot()

for i in range(110):
    ts.run()

ts.pprint(ts.root, 0, 30)