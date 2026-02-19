from src.Bots.mcts import Bot

ts = Bot()

for i in range(100000):
    print(i)
    ts.run()

ts.pprint(ts.root, 0, 100)