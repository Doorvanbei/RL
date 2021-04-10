import numpy as np
import copy
'''
5-by-5 grid world: using DQN ( the Q-DNN is (s,a) --> Q )
author: doorvanbei
date: 20210406
'''
def relu(x):
    return np.maximum(x, 0)
def reluder(x):
    return np.float64(x > 0)
def buildDNN(layersDim): # build a DNN
    DNN = []
    for i in range(len(layersDim)-1):
        w, b = 1.0*(np.random.random((layersDim[i+1],layersDim[i]))-0.5), 0.0*np.random.random((layersDim[i+1],1))
        DNN.append(w), DNN.append(b)
    return DNN
def fp(input,DNN): # forward propagation
    outList = []
    xReluOut = relu(DNN[0] @ input + DNN[1]) # input layer
    outList.append(xReluOut)
    for i in range(1,int(len(DNN)/2)-1): # i = 1,2,3,4
        xReluOut = relu(DNN[2*i] @ xReluOut + DNN[2*i+1])
        outList.append(xReluOut)
    outList.append(DNN[-2] @ xReluOut + DNN[-1])
    return outList
def bp(input,fProcessList,loss): # backward propagation
    dx = 2 * loss / layersDim[-1] / input.shape[1]
    bAid = np.ones((1,input.shape[1]))
    for i in range(1, len(layersDim) - 1):
        dDNN[-2 * i + 1] = lRateDNN * (dx @ bAid.T)
        dDNN[-2 * i] = lRateDNN * (dx @ fProcessList[-i - 1].T)
        dx = (DNN[-2 * i].T @ dx) * reluder(fProcessList[-i - 1])
    dDNN[1] = lRateDNN * (dx @ bAid.T)
    dDNN[0] = lRateDNN * (dx @ input.T)
    for i in range(len(DNN)):
        DNN[i] -= dDNN[i]
def step(s, a): # the env
    return Sp[s, a], R[s, a]
def oneHotVecGen(len,pla): # generate a column one-hot vector
    v = np.zeros((len,1))
    v[pla,0] = 1
    return v
def tPolicy(s):
    inputs = np.concatenate((oneHotVecGen(stateNum, s) @ np.ones((1,actionNum)),np.eye(actionNum)),axis = 0)
    return int(np.argmax(fp(inputs,DNN)[-1],axis=1))
def maxQ(s,DNN):
    inputs = np.concatenate((oneHotVecGen(stateNum, s) @ np.ones((1,actionNum)),np.eye(actionNum)),axis = 0)
    return np.max(fp(inputs, DNN)[-1])
def policy(s,DNN):
    return np.random.randint(0, 4) if np.random.random() < epsilon else tPolicy(s)
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
stateNum = 19
actionNum = 4
inputUnits = stateNum + actionNum
layersDim = [inputUnits,inputUnits,1] # only one dense layer with relu and a output layer
DNN = buildDNN(layersDim)
dDNN = copy.deepcopy(DNN)
DNNTarget = copy.deepcopy(DNN)

iniState,endState = 8,19
lRateDNN = 0.1
epsilon = 0.5  # a small epsilon (small rand) is not beneficial to find global optimal
discountF = 0.9  # discount factor
targetUpdRate = 0.1

pBuffer = 0  # pointer
bufferSize = 2 ** 7
mBatchSize = 64
stateBuffer = np.zeros((bufferSize,),dtype = np.int32)
actionBuffer = np.zeros((bufferSize,),dtype = np.int32)
nStateBuffer = np.zeros((bufferSize,),dtype = np.int32)
rewardBuffer = np.zeros((bufferSize,),dtype = np.float64)
# before training, fill the buffer
while pBuffer != bufferSize: # collect experience until the buffer is full
    state = iniState
    while state != endState:  # go on until the endState is reached
        stateBuffer[pBuffer] = state
        action = policy(state,DNN)
        actionBuffer[pBuffer] = action
        nState, reward = step(state, action)  # step forward in the environment
        nStateBuffer[pBuffer] = nState
        rewardBuffer[pBuffer] = reward
        pBuffer += 1
        if pBuffer != bufferSize:
            state = nState
        else: # the buffer is full
            break

inputs = np.zeros((inputUnits,mBatchSize))
qLabel = np.zeros((mBatchSize,))
pBuffer = 0
for episode in range(100): # each episode is from iniState to endState
    state = iniState
    while state != endState: # each step
        stateBuffer[pBuffer] = state
        action = policy(state,DNN)
        actionBuffer[pBuffer] = action
        nState, reward = step(state, action)  # step forward in the environment
        nStateBuffer[pBuffer] = nState
        rewardBuffer[pBuffer] = reward
        mBatchIdx = np.random.randint(0,bufferSize,(mBatchSize,)) # randomly sample a mini-batch from experience buffer
        mStates,mNStates,mRewards,mActions = stateBuffer[mBatchIdx],nStateBuffer[mBatchIdx],rewardBuffer[mBatchIdx],actionBuffer[mBatchIdx]
        for i in range(mBatchSize): # create qLabel, prepare DNN training
            qLabel[i] = mRewards[i] if mNStates[i] == 19 else mRewards[i] + discountF * maxQ(mNStates[i],DNNTarget)
            inputs[:,i] = np.concatenate((oneHotVecGen(stateNum, mStates[i]), oneHotVecGen(actionNum, mActions[i])),axis=0).reshape((-1,))
        outList = fp(inputs,DNN)
        bp(inputs, outList, outList[-1] - qLabel.reshape((1,-1))) # upd DNN
        for i in range(len(DNN)): # soft upd target DNN
            DNNTarget[i] = (1 - targetUpdRate) * DNNTarget[i] + targetUpdRate * DNN[i]
        pBuffer = pBuffer + 1 if pBuffer != bufferSize - 1 else 0 # circulate pointer with the old experience be covered firstly
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