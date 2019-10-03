import math
import numpy as np


class MetaEnvironment:
    def __init__(self, server_num):
        self.server_num = server_num
        self.traceDict = dict()
        self.nodes = []
        self.accessFreqList = []
        self.totalAccessFreqList = []
        self.loadList = []
        self.actionRecord = []
        self.hotNodeList = []
        self.hotNodeVecMatrix = []
        self.limit = 0

        f = open("createfile.txt")
        # 静态子树初始化
        for trace in f.readlines():
            trace = trace.strip()
            if trace == "/ROOT":
                first_layer = ''
            else:
                first_layer = trace.split("/")[2]
            serverID = hash(first_layer) % self.server_num
            pos = len(self.nodes)
            self.traceDict[trace] = [pos, serverID]
            self.nodes.append(trace)

        # 动态子树初始化
        # for trace in f.readlines():
        #     trace = trace.strip()
        #     components = trace.split("/")
        #     if len(components) <= 4:
        #         if trace == "/ROOT":
        #             first_layer = ''
        #         else:
        #             first_layer = trace.split("/")[2]
        #         serverID = hash(first_layer) % self.server_num
        #     elif len(components) > 4 and len(components) <= 7:
        #         serverID = hash(components[4]) % self.server_num
        #     else:
        #         serverID = hash(components[7]) % self.server_num
        #     pos = len(self.nodes)
        #     self.traceDict[trace] = [pos, serverID]
        #     self.nodes.append(trace)

        # Hash初始化
        # for trace in f.readlines():
        #     trace = trace.strip()
        #     serverID = hash(trace) % self.server_num
        #     pos = len(self.nodes)
        #     self.traceDict[trace] = [pos, serverID]
        #     self.nodes.append(trace)
        # f.close()

        f = open("path2coord-hot.txt")
        # f = open("path2coord.txt")
        for line in f.readlines():
            line = line.strip()
            nodeCoord = line.split(" ")
            self.hotNodeList.append(nodeCoord[0])
            self.hotNodeVecMatrix.append(self.handleStr(nodeCoord[1]))
        f.close()
        # self.accessFreqList = [0 for _ in self.nodes]
        self.totalAccessFreqList = [0 for _ in self.nodes]

    def state(self, traceList):
        self.accessFreqList = [0 for _ in self.nodes]
        for trace in traceList:
            components = trace.split("/")
            for index, node in enumerate(components):
                metadata = "/".join(components[:index + 1])
                if metadata == "":
                    continue
                pos = self.traceDict[metadata][0]
                self.accessFreqList[pos] = self.accessFreqList[pos] + 1
                self.totalAccessFreqList[pos] = self.totalAccessFreqList[pos] + 1
        stateMatrix = []
        # print("accessFreq:", self.accessFreqList)
        for index, node in enumerate(self.hotNodeList):
            rowVec = []
            rowVec.extend(self.hotNodeVecMatrix[index])
            serverID = self.traceDict[node][1]
            onehotVec = [0 for _ in range(self.server_num)]
            onehotVec[serverID] = 1
            rowVec.extend(onehotVec)
            rowVec.append(self.accessFreqList[self.traceDict[node][0]])
            stateMatrix.extend(rowVec)

        # for line in f.readlines():
        #     line = line.strip()
        #     rowVec = []
        #     nodeCoord = line.split(" ")
        #     rowVec.extend(self.handleStr(nodeCoord[1]))
        #     serverID = self.traceDict[nodeCoord[0]][1]
        #     onehotVec = [0 for _ in range(self.server_num)]
        #     onehotVec[serverID] = 1
        #     rowVec.extend(onehotVec)
        #     rowVec.append(self.accessFreqList[self.traceDict[nodeCoord[0]][0]])
        #     stateMatrix.append(rowVec)

        arr = np.array(stateMatrix)
        arr = arr.reshape(-1, len(stateMatrix))
        return arr

    def take_actions(self, moveList):
        # if self.limit < 10:
        #     self.limit += 1
        #     return 0

        migCnt = 0
        moveList = moveList.tolist()
        # print("raw-movelist", moveList)
        moveList = [math.floor(x * self.server_num) for x in moveList]
        # moveList = [round(x) % self.server_num for x in moveList]
        # print("movelist", moveList)
        for index, serverId in enumerate(moveList):
            metadata = self.hotNodeList[index]
            if serverId == self.server_num:
                serverId -= 1
            if serverId != self.traceDict[metadata][1]:
                self.traceDict[metadata][1] = serverId
                migCnt += 1
        return migCnt

    def locality(self):
        Loc = 0
        for metadata in self.nodes:
            components = metadata.split("/")
            serverID = -1
            jumps = 0
            for index, _ in enumerate(components):
                metadata = "/".join(components[:index + 1])
                if metadata == "":
                    continue
                pos = self.traceDict[metadata][0]
                if serverID != self.traceDict[metadata][1]:
                    serverID = self.traceDict[metadata][1]
                    jumps = jumps + 1
            # print("locality access", Loc, self.accessFreqList)
            Loc = Loc + jumps * self.accessFreqList[pos]
        Loc = 1 / float(Loc)

        return Loc

    def load(self):
        self.loadList = [0 for _ in range(self.server_num)]
        for _, value in self.traceDict.items():
            self.loadList[value[1]] = self.loadList[value[1]] + self.accessFreqList[value[0]]
        avg_load = float(sum(self.loadList)) / self.server_num
        Bal = 0
        print("load list:", self.loadList)
        for load in self.loadList:
            Bal = Bal + math.pow(load - avg_load, 2)
        Bal = (self.server_num - 1) / float(Bal)

        return Bal

    def final_locality(self):
        Loc = 0
        for metadata in self.nodes:
            components = metadata.split("/")
            serverID = -1
            jumps = 0
            for index, _ in enumerate(components):
                metadata = "/".join(components[:index + 1])
                if metadata == "":
                    continue
                pos = self.traceDict[metadata][0]
                if serverID != self.traceDict[metadata][1]:
                    serverID = self.traceDict[metadata][1]
                    jumps = jumps + 1
            # print("locality access", Loc, self.accessFreqList)
            Loc = Loc + jumps * self.totalAccessFreqList[pos]
        Loc = 1 / float(Loc)

        return Loc

    def final_load(self):
        self.loadList = [0 for _ in range(self.server_num)]
        for _, value in self.traceDict.items():
            self.loadList[value[1]] = self.loadList[value[1]] + self.totalAccessFreqList[value[0]]
        avg_load = float(sum(self.loadList)) / self.server_num
        Bal = 0
        print("load list:", self.loadList)
        for load in self.loadList:
            Bal = Bal + math.pow(load - avg_load, 2)
        Bal = (self.server_num - 1) / float(Bal)

        return Bal

    def clear(self):
        self.totalAccessFreqList = [0 for _ in range(len(self.nodes))]

    def handleStr(self, str):
        str = str[:-1].split(",")
        l = [int(x) for x in str]
        return l
