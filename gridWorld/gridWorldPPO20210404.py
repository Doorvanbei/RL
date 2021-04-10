import numpy as np
import copy
'''
5-by-5 grid world: using PPO
author: doorvanbei
date: 20210404
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
    l = len(DNN)
    denseTimes = int((l-1)/2) - 1
    batchSize = inputs.shape[1]
    outList = []
    bAid = np.ones((1, batchSize), dtype=np.float64)
    x = inputs
    for i in range(0,denseTimes+1):
        x = relu(DNN[2*i] @ x + DNN[2*i+1] @ bAid)
        outList.append(x)
    x = softmax(DNN[-1] @ x)
    outList.append(x)
    return outList
def bpActor(inputs,outList,dBefSoft):
    l = len(actorDNN)
    lout = len(outList)
    denseTimes = int((l-1) / 2) - 1
    batchSize = inputs.shape[1]
    bAid = np.ones((1, batchSize), dtype=np.float64)
    dx = dBefSoft
    dActorDNN[l-1] = dx @ outList[lout - 2].T # W
    dx = actorDNN[l-1].T @ dx * reluder(outList[lout - 2])
    for i in range(1,1+denseTimes):
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
def policy(s): # return the action and its possibility from where it is selected
    pVector = fpActor(oneHotVecGen(actorLayersDim[0], s), actorDNN)[-1]
    action = np.where(np.cumsum(pVector) - np.random.uniform() > 0)[0][0]
    return action, np.float64(pVector[action])
def step(s, a):
    return Sp[s, a], R[s, a]
def playOnce():
    stateListPerGame, actionListPerGame, pActionListPerGame, rewardListPerGame = np.array([], dtype=np.int32), np.array([],dtype=np.int32), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    state = iniState
    stateListPerGame = np.append(stateListPerGame, state)
    action, possibility = policy(state)
    actionListPerGame,pActionListPerGame = np.append(actionListPerGame, action), np.append(pActionListPerGame, possibility)
    nState, reward = step(state, action)
    while nState != 19:  # play game forwardly
        rewardListPerGame = np.append(rewardListPerGame, reward)
        state = nState
        stateListPerGame = np.append(stateListPerGame, state)
        action, possibility = policy(state)
        actionListPerGame, pActionListPerGame = np.append(actionListPerGame, action), np.append(pActionListPerGame, possibility)
        nState, reward = step(state, action)
    rewardListPerGame = np.append(rewardListPerGame, reward)
    inputsCritic = np.zeros((criticLayersDim[0], stateListPerGame.size))
    inputsCritic = mark(inputsCritic, stateListPerGame, stateListPerGame.size)
    vOfStates = fpCritic(inputsCritic, criticDNN)[-1].reshape((-1,))
    vNext = np.zeros_like(vOfStates)
    vNext[:-1] = vOfStates[1:]
    return stateListPerGame,actionListPerGame,pActionListPerGame,rewardListPerGame + discountF * vNext
def checkPlay(): # return the trajectory once the training is over.
    traj = []
    traj.append(iniState)
    a,_ = policy(iniState)
    nState,_ = step(iniState,a)
    while nState != 19:
        traj.append(nState)
        a,_ = policy(nState)
        nState,_ = step(nState,a)
    return traj
def sampleMiniBatch(states, actions, possibilities, gValues):
    batchSize = states.size
    sampleArray = np.random.randint(0,batchSize,(expMiniBatch,))
    return states[sampleArray], actions[sampleArray], possibilities[sampleArray], gValues[sampleArray]
def checkPlay(): # return the trajectory once the training is over.
    traj = []
    traj.append(iniState)
    action,_ = policy(iniState)
    nState,_ = step(iniState,action)
    while nState != 19:
        traj.append(nState)
        action,_ = policy(nState)
        nState,_ = step(nState,action)
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
clipF = 0.1
gamesPerBatch = 100
expMiniBatch = 128 # sample this number of experiences as a mini-batch to start train DNNs
learnRateActorDNN = 1
learnRateCriticDNN = 1
iniState = 1
actorLayersDim,criticLayersDim = [19,19,19,19,4],[19,19,19,19,1]
actorDNN, criticDNN = buildActorDNN(actorLayersDim), buildCriticDNN(criticLayersDim)
dActorDNN, dCriticDNN = copy.deepcopy(actorDNN), copy.deepcopy(criticDNN)
for interacts in range(6):
    print(f'train epoch = {interacts}')
    states, actions, possibilities, gValues = np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([],dtype=np.float64), np.array([], dtype=np.float64)
    for games in range(500):
        sPerGame,aPerGame,pPerGame,gPerGame = playOnce() # get 3 np-1d-array
        states, actions, possibilities, gValues = np.append(states,sPerGame), np.append(actions,aPerGame), np.append(possibilities,pPerGame), np.append(gValues,gPerGame)
    for mBatchTrain in range(20):
        mStates, mActions, mPossibilities, mGValues = sampleMiniBatch(states, actions, possibilities, gValues)
        mInputs = np.zeros((actorLayersDim[0],expMiniBatch))
        mInputs = mark(mInputs,mStates,expMiniBatch)
        mGValues = mGValues.reshape((1,-1))
        dValue = (mGValues - fpCritic(mInputs, criticDNN)[-1]).reshape((-1,)) # advantage function
        for trainC in range(100):    # train critic
            outListC = fpCritic(mInputs, criticDNN)
            out = outListC[-1]
            errV = out - mGValues
            lossC = np.float64(errV @ errV.T / expMiniBatch) # train critic to reduce this loss
            # print(lossC)
            bpCritic(mInputs,outListC,2*errV/expMiniBatch)
        for trainA in range(50):    # train actor
            outListA = fpActor(mInputs, actorDNN)
            out = outListA[-1]
            mLabel = np.zeros_like(out)
            mLabel = mark(mLabel, mActions, expMiniBatch)
            p = out[mActions,np.arange(expMiniBatch)]
            r = p / mPossibilities
            c = np.maximum(np.minimum(r, 1.0+clipF), 1.0-clipF)
            objectiveA = np.mean(np.minimum(dValue*r,dValue*c)) # train actor to increase this objective
            # print(objectiveA)
            dr = dValue/expMiniBatch
            dr[((dValue > 0) & (r > 1+clipF)) | ((dValue < 0) & (r < 1-clipF))] = 0 # der of clip
            dp = dr / mPossibilities
            dp = np.ones((actorLayersDim[-1],1)) @ dp.reshape((1,-1))
            p = np.ones((actorLayersDim[-1],1)) @ p.reshape((1,-1))
            bpActor(mInputs,outListA,p * (mLabel - out) * dp)
# test final result
inputs = np.eye(19)
outA = fpActor(inputs, actorDNN)[-1]
outC = fpCritic(inputs, criticDNN)[-1]
for i in range(10):
    print(checkPlay())
