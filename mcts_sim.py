from src.Bots.mcts import Bot

ts = Bot()
ts.load_tree()
ts.pprint(ts.root, 0, 100)

for i in range(100000):
    print(i)
    ts.run()

ts.pprint(ts.root, 0, 100)
ts.save_tree()