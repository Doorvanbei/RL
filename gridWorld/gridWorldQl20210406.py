import numpy as np
'''
5-by-5 grid world: using Q-learning
author: doorvanbei
date: 20210406
'''
def policy(s): # learning policy
    return np.random.randint(0,4) if np.random.random() < epsilon else np.argmax(qTable[s])
def tPolicy(s): # test policy / the policy adopted in practical applications
    return np.argmax(qTable[s])
def step(s, a): # the environment
    return Sp[s, a], R[s, a]
R = np.array([[-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, 5, -1],
              [-1, -1, -1, -1],
              [-1, -1, -1, 5],
              [-1, -1, -1, -1],
              [-1, -1, 10, -1],
              [-1, -1, -1, -1],
              [5, -1, -1, -1],
              [-1, -1, -1, 10],
              ], dtype=np.float64) # is a part of environment
Sp = np.array([[0, 0, 5, 1],
               [1, 0, 6, 2],
               [2, 1, 7, 3],
               [3, 2, 8, 4],
               [4, 3, 9, 4],
               [0, 5, 10, 6],
               [1, 5, 11, 7],
               [2, 6, 7, 8],
               [3, 7, 8, 9],
               [4, 8, 12, 9],
               [5, 10, 13, 11],
               [6, 10, 14, 11],
               [9, 12, 15, 12],
               [10, 13, 16, 14],
               [14, 14, 18, 15],
               [12, 14, 19, 15],
               [13, 16, 16, 17],
               [14, 16, 17, 17],
               [14, 18, 18, 19]], dtype=np.int32) # is a part of environment

np.random.seed(1)
iniState,endState = 8,19
epsilon = 0.5  # possibility to choose a random action
discountF = 0.9 # discount factor
updRate = 0.1 # update rate
qTable = np.random.random((19, 4)) # initialize Q-table
# train
for episode in range(1000):# each episode is a trajectory from initial state to the end state
    state = iniState
    while state != endState: # go on until the endState is reached
        action = policy(state) # choose an action use learning policy
        nState, reward = step(state,action) # step forward in the environment
        qLabel = reward if nState == endState else reward + discountF * qTable[nState, tPolicy(nState)] # make q value label
        qTable[state, action] = (1 - updRate) * qTable[state, action] + updRate * qLabel # update q(S,A) using its old value and q value label
        state = nState
# test
state = iniState
stateList = [state]
while state != endState:
    action = tPolicy(state) # choose an action use test policy
    nState, reward = step(state,action)
    state = nState
    stateList.append(state)
print(stateList)
