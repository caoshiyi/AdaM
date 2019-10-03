import random
from Environment import MetaEnvironment as Env

f = open("query.txt")

queryList = []
for line in f.readlines():
    line = line.strip()
    queryList.append(line)

env = Env(5)

for i in range(5):
    traceList = queryList[i * 10:(i + 1) * 10]
    state = env.state(traceList)
    print(state)
    moveList = [random.randint(0, env.server_num - 1) for _ in range(len(env.nodes))]
    env.take_actions(moveList)
    print('Loc:', env.locality())
    print('Load:', env.load())
