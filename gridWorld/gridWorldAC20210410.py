import numpy as np
import copy
'''
5-by-5 grid world: using Actor-Critic
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
def buildActorDNN(layersDim):
    DNN = []
    for i in range(len(layersDim)-1):
        w, b = np.random.random((layersDim[i+1],layersDim[i]))-0.5, 0.01*(np.random.random((layersDim[i+1],1)) - 0.5)
        DNN.append(w), DNN.append(b)
    DNN.pop()
    return DNN
def buildCriticDNN(layersDim): # build a DNN
    DNN = []
    for i in range(len(layersDim)-1):
        w, b = 1.0*(np.random.random((layersDim[i+1],layersDim[i]))-0.5), 0.0*np.random.random((layersDim[i+1],1))
        DNN.append(w), DNN.append(b)
    return DNN
def fpActor(inputs,DNN):
    l = len(DNN) # l = 9
    DenseTimes = int((l-1)/2) - 1 # 1
    batchSize = inputs.shape[1]
    outList = []
    bAid = np.ones((1, batchSize), dtype=np.float64)
    x = inputs
    for i in range(0,DenseTimes+1):
        x = relu(DNN[2*i] @ x + DNN[2*i+1] @ bAid)
        outList.append(x)
    x = softmax(DNN[-1] @ x)
    outList.append(x)
    return outList
def bpActor(inputs,outList,dBefSoft):
    l = len(actorDNN)
    lout = len(outList)
    DenseTimes = int((l-1) / 2) - 1
    batchSize = inputs.shape[1]
    bAid = np.ones((1, batchSize), dtype=np.float64)
    dx = dBefSoft
    dActorDNN[l-1] = dx @ outList[lout - 2].T # W
    dx = actorDNN[l-1].T @ dx * reluder(outList[lout - 2])
    for i in range(1,1+DenseTimes):
        dActorDNN[l-1-2*i] = dx @ outList[lout - 2 - i].T
        dActorDNN[l-2*i] = dx @ bAid.T
        dx = actorDNN[l-1-2*i].T @ dx * reluder(outList[lout - 2 - i])
    dActorDNN[0] = dx @ inputs.T
    dActorDNN[1] = dx @ bAid.T
    for i in range(l):
        actorDNN[i] += learnRateActorDNN * dActorDNN[i] # gradient ascent
def fpCritic(inputs,DNN): # forward propagation
    outList = []
    batchSize = inputs.shape[1]
    bAid = np.ones((1,batchSize))
    xReluOut = relu(DNN[0] @ inputs + DNN[1] @ bAid)
    outList.append(xReluOut)
    for i in range(1,int(len(DNN)/2)-1): # i = 1,2,3,4
        xReluOut = relu(DNN[2*i] @ xReluOut + DNN[2*i+1] @ bAid)
        outList.append(xReluOut)
    xOut = DNN[-2] @ xReluOut + DNN[-1] @ bAid
    outList.append(xOut)
    return outList
def bpCritic(inputs,outList,loss): # backward propagation
    dx = 2 * loss / criticLayersDim[-1] / inputs.shape[1]
    bAid = np.ones((1, inputs.shape[1]), dtype=np.float64)
    for i in range(1, len(criticLayersDim) - 1):
        dCriticDNN[-2 * i + 1] = learnRateCriticDNN * (dx @ bAid.T)
        dCriticDNN[-2 * i] = learnRateCriticDNN * (dx @ outList[-i - 1].T)
        dx = (criticDNN[-2 * i].T @ dx) * reluder(outList[-i - 1])
    dCriticDNN[1] = learnRateCriticDNN * (dx @ bAid.T)
    dCriticDNN[0] = learnRateCriticDNN * (dx @ inputs.T)
    for i in range(len(criticDNN)):
        criticDNN[i] -= dCriticDNN[i]
def oneHotVecGen(len,pla):
    v = np.zeros((len,1))
    v[pla,0] = 1
    return v
def policy(s):
    return np.where(np.cumsum(fpActor(oneHotVecGen(actorLayersDim[0], s), actorDNN)[-1]) - np.random.uniform() > 0)[0][0]
def step(s, a):
    return Sp[s, a], R[s, a]
def playOnce():
    stateListPerGame, actionListPerGame, rewardListPerGame = np.array([], dtype=np.int32), np.array([],dtype=np.int32), np.array([], dtype=np.float64)
    state = iniState
    stateListPerGame = np.append(stateListPerGame, state)
    action = policy(state)
    actionListPerGame = np.append(actionListPerGame, action)
    nState, reward = step(state, action)
    while nState != 19:  # play game forwardly
        rewardListPerGame = np.append(rewardListPerGame, reward)
        state = nState
        stateListPerGame = np.append(stateListPerGame, state)
        action = policy(state)
        actionListPerGame = np.append(actionListPerGame, action)
        nState, reward = step(state, action)
    rewardListPerGame = np.append(rewardListPerGame, reward)
    inputsCritic = np.zeros((criticLayersDim[0], stateListPerGame.size))
    inputsCritic = mark(inputsCritic, stateListPerGame, stateListPerGame.size)
    vOfStates = fpCritic(inputsCritic, criticDNN)[-1].reshape((-1,))
    vNext = np.zeros_like(vOfStates)
    vNext[:-1] = vOfStates[1:]
    return stateListPerGame,actionListPerGame,rewardListPerGame + discountF * vNext - vOfStates
def checkPlay(): # return the trajectory once the training is over.
    traj = []
    traj.append(iniState)
    nState,_ = step(iniState,policy(iniState))
    while nState != 19:
        traj.append(nState)
        nState,_ = step(nState,policy(nState))
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
discountF = 1
learnRateActorDNN = 0.1
learnRateCriticDNN = 0.1
iniState = 1
actorLayersDim,criticLayersDim = [19,19,4],[19,19,1]
actorDNN, criticDNN = buildActorDNN(actorLayersDim), buildCriticDNN(criticLayersDim)
dActorDNN, dCriticDNN = copy.deepcopy(actorDNN), copy.deepcopy(criticDNN)
inputs, labelA, weight, lossC = np.array([],dtype=np.float64),np.array([],dtype=np.float64),np.array([],dtype=np.float64),np.array([],dtype=np.float64)
for trains in range(500):
    print(f'train epoch = {trains}')
    sPerGame,aPerGame,dPerGame = playOnce() # get 3 np-1d-array
    batch = sPerGame.size
    inputThisGame, labelThisGame = np.zeros((actorLayersDim[0], batch)), np.zeros((actorLayersDim[-1], batch))
    inputThisGame, labelThisGame = mark(inputThisGame, sPerGame, batch), mark(labelThisGame, aPerGame, batch)
    weightThisGame = np.ones((actorLayersDim[-1],1)) @ dPerGame.reshape((1,batch))
    inputs, labelA, weight, lossC = inputThisGame, labelThisGame, weightThisGame, dPerGame.reshape((1,batch))
    outListA = fpActor(inputs,actorDNN)
    out = outListA[-1]
    bpActor(inputs, outListA, (labelA - out) * weight / inputs.shape[-1])
    outListC = fpCritic(inputs, criticDNN)
    out = outListC[-1]
    bpCritic(inputs,outListC,-2*lossC / inputs.shape[-1])
# check results after training
inputs = np.eye(19)
outA = fpActor(inputs,actorDNN)[-1]
outC = fpCritic(inputs,criticDNN)[-1]
for i in range(10):
    print(checkPlay())
