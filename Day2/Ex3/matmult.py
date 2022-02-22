# Program to multiply two matrices using nested loops
import random
import numpy as np #with numpy it works 25 times (approx) faster

N = 250
    
@profile    
def perform_mult():
    # NxN matrix
    X = []
    for i in range(N):
        X.append([random.randint(0,100) for r in range(N)])
    
    # Nx(N+1) matrix
    Y = []
    for i in range(N):
        Y.append([random.randint(0,100) for r in range(N+1)])
    
    # result is Nx(N+1)
    result = []
    for i in range(N):
        result.append([0] * (N+1))
        
    # create numpy arrays
    np_X = np.array(X)
    np_Y = np.array(Y)
    np_result = np.array(result)
    
    nX = len(np_X)
    nY0 = len(np_Y[0])
    nY = len(np_Y)
    # iterate through rows of X
    for i in range(nX):
        # iterate through columns of Y
        for j in range(nY0):
            result[i][j] = np.dot(np_X[i], np_Y[:,j])
    
    for r in result:
        print(r)
    
perform_mult()