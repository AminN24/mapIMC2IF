#!/home/amin/miniconda3/bin/python

from gurobipy import *
import numpy as np
import pandas as pd
import pickle
import sys

class Matching:
    def __init__(self, P, dist, r):
        self.P = P
        self.dist = dist
        self.r = r
        (M, N) = P.shape
        self.M = M
        self.N = N
        self.x = [[0 for j in range(N)] for i in range(M)]

        self.model = Model("IMC-IF-Assignment-Problem")

    def formulate(self):
        
        # 1) Define variables
        for i in range(self.M):
            for j in range(self.N):
                self.x[i][j] = self.model.addVar(vtype=GRB.BINARY,
                                                 name=f"x_{i}_{j}")

        # 2) Define constraints
        # -- Unique mapping
        for i in range(self.M):
            self.model.addConstr(quicksum(self.x[i]), GRB.LESS_EQUAL, 1,
                                 name=f"unique_matching_M_{i}")
            
        for j in range(self.N):
            self.model.addConstr(
                quicksum([self.x[i][j] for i in range(self.M)]),
                GRB.LESS_EQUAL, 1, name=f"unique_matching_N_{j}"
            )
        
        # -- Distance constraint
        for i in range(self.M):
            for j in range(self.N):
                if self.dist[i, j] > self.r:
                    self.model.addConstr(self.x[i][j], GRB.EQUAL, 0,
                                         name=f"distance_{i}_{j}")

        # 3) Define objective
        self.model.setObjective(
            quicksum([self.x[i][j] * self.P[i, j]
                      for i in range(self.M) for j in range(self.N)]),
            GRB.MAXIMIZE
        )

    def match(self):
        self.formulate()
        self.model.update()
        try:
            self.model.optimize()
        except GurobiError:
            print("Optimize failed due to non-convexity")
        #self.model.write('matching.lp')

    def getResults(self):
        matchings = []
        for i in range(self.M):
            for j in range(self.N):
                if self.x[i][j].X == 1:
                    matchings.append([i + 1, j + 1, self.dist[i, j]])
                   # print(self.x[i][j].VarName)
        
        #matchings = pd.DataFrame(matchings, columns=['X_i', 'Y_i'])
        return(matchings)


def loadData(path):
    """Load any data in .pickle format."""
    f = open(path, "rb")
    data = pickle.load(f)
    print(f"Data loaded from {path}!")
    return(data)


def pointsetDistance(X, Y):
    Y = np.repeat(Y[:, np.newaxis, :], len(X), axis=1)
    X = np.repeat(X[np.newaxis, :, :], len(Y), axis=0)
    dist = Y - X
    dist = np.sqrt(np.sum(dist ** 2, axis=2))
    return(dist)


def main():
    input_path = sys.argv[1]
    radius = 15
    data = loadData(input_path)
    Yt = data['Yt']
    P = data['P']
    imc_coords = data['imc_coords']
    if_coords = data['if_coords']
    print(imc_coords.head())
    print(if_coords.head())
    if len(imc_coords) == len(Yt):
        X = if_coords.loc[:,['x_n', 'y_n']].to_numpy()
        X_name = 'IF'
    elif len(if_coords) == len(Yt):
        X = imc_coords.loc[:, ['x_n', 'y_n']].to_numpy()
        X_name = 'IMC'
    print("X_name: ", X_name)
    dist = pointsetDistance(X=X, Y=Yt)
    
    print("Inputs:")
    print("Distance matrix:", dist.shape, dist.min(), dist.max())
    print("Probability matrix:", P.shape, P.min(), P.max())
    print("Adjacency threshold (radius):", radius)
    
    assert dist.shape == P.shape

    print("Matching started...")
    matching = Matching(P=P, dist=dist, r=radius)
    matching.match()
    print("Matching succeeded!")
    result = matching.getResults()

    if X_name == "IMC":
        columns=['if_cell_id', 'imc_cell_id', 'distance']
    elif X_name == "IF":
        columns=['imc_cell_id', 'if_cell_id', 'distance']

    output = pd.DataFrame(result, columns=columns)

    print(output.head())
    output.to_csv(input_path+".txt", sep="\t", index=False)
    
if __name__ == "__main__":
    main()
