import numpy as np
import copy
'''
5-by-5 grid world: using Policy Gradient 
author: doorvanbei
date: 20210410
'''
def relu(x):
    return np.maximum(x, 0)
def reluder(x):
    return np.float64(x > 0)
def softmax(x): # input a column vector or multi-column matrix where each column is a sample
    # X = x - np.max(x,axis=0)
    out = np.exp(x)
    l = out.shape[0]
    summ = np.ones((l,l)) @ out
    out /= summ
    return out
def mark(x,l,batchSize):
    X = x
    X[l,np.arange(0,batchSize)] = 1.0
    return X
def buildDNN(layersDim): # build a DNN
    DNN = []
    for i in range(len(layersDim)-1):
        w, b = np.random.random((layersDim[i+1],layersDim[i]))-0.5, 0.01*(np.random.random((layersDim[i+1],1)) - 0.5)
        DNN.append(w), DNN.append(b)
    DNN.pop()
    return DNN
def fp(inputs):
    l = len(DNN) # l = 9
    DenseTimes = int((l-1)/2) - 1 # 1
    batchSize = inputs.shape[1]
    outList = []
    bAid = np.ones((1, batchSize), dtype=np.float64)
    x = inputs
    for i in range(0,DenseTimes+1):
        x = relu(DNN[2*i] @ x + DNN[2*i+1] @ bAid)
        outList.append(x)
    # output
    x = softmax(DNN[-1] @ x)
    outList.append(x)
    return outList # [0,1,2]
def bp(inputs,outList,dBefSoft):
    l = len(DNN) # l = 5
    lout = len(outList) # 3
    DenseTimes = int((l-1) / 2) - 1 # 1
    batchSize = inputs.shape[1]
    bAid = np.ones((1, batchSize), dtype=np.float64)
    dx = dBefSoft
    dDNN[l-1] = dx @ outList[lout - 2].T # W
    dx = DNN[l-1].T @ dx * reluder(outList[lout - 2])
    for i in range(1,1+DenseTimes):
        dDNN[l-1-2*i] = dx @ outList[lout - 2 - i].T
        dDNN[l-2*i] = dx @ bAid.T
        dx = DNN[l-1-2*i].T @ dx * reluder(outList[lout - 2 - i])
    dDNN[0] = dx @ inputs.T
    dDNN[1] = dx @ bAid.T
    for i in range(l):
        DNN[i] += learnRateDNN * dDNN[i] # gradient ascent
def step(s, a): # the env
    return Sp[s, a], R[s, a]
def oneHotVecGen(len,pla): # generate a column one-hot vector
    v = np.zeros((len,1))
    v[pla,0] = 1
    return v
def policy(s,epsi): # agent give an action according to its policy
    if np.random.random() > epsi: # epsi get smaller with training time grows
        return np.where(np.cumsum(fp(oneHotVecGen(layersDim[0], s))[-1]) - np.random.uniform() > 0)[0][0]
    else: # do not run this block in this policy gradient program.
        return np.random.randint(0,layersDim[-1])
def playOnce():
    stateListPerGame, actionListPerGame, rewardListPerGame = np.array([], dtype=np.int32), np.array([],dtype=np.int32), np.array([], dtype=np.float64)
    state = iniState
    stateListPerGame = np.append(stateListPerGame, state)
    epsi = 0
    action = policy(state, epsi)
    actionListPerGame = np.append(actionListPerGame, action)
    nState, reward = step(state, action)
    while nState != 19:  # play game forwardly
        rewardListPerGame = np.append(rewardListPerGame, reward)
        state = nState
        stateListPerGame = np.append(stateListPerGame, state)
        action = policy(state, epsi)
        actionListPerGame = np.append(actionListPerGame, action)
        nState, reward = step(state, action)
    rewardListPerGame = np.append(rewardListPerGame, reward)
    return stateListPerGame,actionListPerGame,np.sum(rewardListPerGame)
def checkPlay(): # return the trajectory once the training is over.
    traj = []
    traj.append(iniState)
    nState,_ = step(iniState,policy(iniState,0))
    while nState != 19:
        traj.append(nState)
        nState,_ = step(nState,policy(nState,0))
    return traj
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
              ], dtype=np.float64) # R: Reward matrix, global
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
               [14, 18, 18, 19]], dtype=np.int32) # Sp: State transition matrix, global

np.random.seed(1)
iniState = 1
learnRateDNN = 0.05 # 0.005 is the initial learnRate for DNN, tune the exponent 0 up with training times
layersDim = [19,19,4] # input num, dense units, ..., befSoftmaxUnits, at least one Dense unit (at least len = 3)
gamesPerBatch = 1 # N
trainDNNTPB = 1 # do not need multiple training on one batch, since this is policy gradient

DNN = buildDNN(layersDim)
dDNN = copy.deepcopy(DNN)

inputs, label, weight = np.array([],dtype=np.float64),np.array([],dtype=np.float64),np.array([],dtype=np.float64)
# start training
for training in range(500): # after each loop, tune down the learnRateDNN
    stateListPerGame,actionListPerGame,RPerGame = playOnce()
    batch = stateListPerGame.size
    inputThisGame,labelThisGame = np.zeros((layersDim[0],batch)),np.zeros((layersDim[-1],batch))
    inputThisGame,labelThisGame = mark(inputThisGame, stateListPerGame, batch),mark(labelThisGame,actionListPerGame,batch)
    weightThisGame = RPerGame*np.ones((layersDim[-1],batch),dtype=np.float64)
    inputs,label,weight = inputThisGame,labelThisGame,weightThisGame
    outList = fp(inputs)
    out = outList[-1]
    bp(inputs,outList,(label - out)*weight/inputs.shape[-1])
    J = np.sum(np.log(out)*label*weight) # objective function
    print(f'Avg reward in batch {training:5} is {RPerGame:7.3f} and J is {J:10.5f}')

inputs = np.eye(layersDim[0]) # generate all kinds of states
out = fp(inputs)[-1] # observe the policy under all kinds of states after training
print(checkPlay()) # check the policy after training
