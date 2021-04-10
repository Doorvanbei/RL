import numpy as np
import copy
import gym
import time
'''
Pendulum-v0: td3 (DNN,DNN2 -> critic, aDNN -> actor)
author: doorvanbei
date: 202104010
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
def fp(inputs,DNN): # forward propagation
    outList = []
    xReluOut = relu(DNN[0] @ inputs + DNN[1]) # inputs layer
    outList.append(xReluOut)
    for i in range(1,int(len(DNN)/2)-1): # i = 1,2,3,4
        xReluOut = relu(DNN[2*i] @ xReluOut + DNN[2*i+1])
        outList.append(xReluOut)
    outList.append(DNN[-2] @ xReluOut + DNN[-1])
    return outList
def bp(inputs,outList,dOut): # backward propagation
    dx = copy.deepcopy(dOut)
    bAid = np.ones((1,inputs.shape[1]))
    for i in range(1, len(layersDim) - 1):
        dDNN[-2 * i + 1] = dx @ bAid.T
        dDNN[-2 * i] = dx @ outList[-i - 1].T
        dx = (DNN[-2 * i].T @ dx) * reluder(outList[-i - 1])
    dDNN[1] = dx @ bAid.T
    dDNN[0] = dx @ inputs.T
    for i in range(len(DNN)):
        DNN[i] -= lRateDNN * dDNN[i]
def bp2(inputs,outList,dOut): # backward propagation
    dx = copy.deepcopy(dOut)
    bAid = np.ones((1,inputs.shape[1]))
    for i in range(1, len(layersDim) - 1):
        dDNN2[-2 * i + 1] = dx @ bAid.T
        dDNN2[-2 * i] = dx @ outList[-i - 1].T
        dx = (DNN2[-2 * i].T @ dx) * reluder(outList[-i - 1])
    dDNN2[1] = dx @ bAid.T
    dDNN2[0] = dx @ inputs.T
    for i in range(len(DNN2)):
        DNN2[i] -= lRateDNN * dDNN2[i]
def bpNUPD(outList,dOut): # backward propagation
    dx = copy.deepcopy(dOut)
    for i in range(1, len(layersDim) - 1):
        dx = (DNN[-2 * i].T @ dx) * reluder(outList[-i - 1])
    return DNN[0].T @ dx
def bpAdNN(inputs,outList,dOut): # backward propagation
    dx = copy.deepcopy(dOut)
    bAid = np.ones((1,inputs.shape[1]))
    for i in range(1, len(aLayersDim) - 1):
        dADNN[-2 * i + 1] = dx @ bAid.T
        dADNN[-2 * i] = dx @ outList[-i - 1].T
        dx = (AdNN[-2 * i].T @ dx) * reluder(outList[-i - 1])
    dADNN[1] = dx @ bAid.T
    dADNN[0] = dx @ inputs.T
    for i in range(len(AdNN)):
        AdNN[i] -= lRateAdNN * dADNN[i]
def oneHotVecGen(len,pla): # generate a column one-hot vector
    v = np.zeros((len,1))
    v[pla,0] = 1
    return v
def tPolicy(s,DNN):
    return fp(s,DNN)[-1]
def policy(s,DNN):
    return np.random.uniform(-2,2,(1,1)) if np.random.random() < epsilon else tPolicy(s,DNN)
np.random.seed(2)
env = gym.make('Pendulum-v0')
env.seed(2)

stateNum = 3
actionNum = 1
inputUnits = stateNum + actionNum
layersDim = [inputUnits,64,1] # only one dense layer with relu and a output layer
aLayersDim = [stateNum,64,1]
DNN,DNN2,AdNN = buildDNN(layersDim),buildDNN(layersDim),buildDNN(aLayersDim)
dDNN,DNNTarget = copy.deepcopy(DNN),copy.deepcopy(DNN)
dDNN2,DNN2Target = copy.deepcopy(DNN2),copy.deepcopy(DNN2)
dADNN,AdNNTarget = copy.deepcopy(AdNN),copy.deepcopy(AdNN)
lRateDNN = 0.01 # 0.05
lRateAdNN = 0.005 # 0.3
epsilon = 0.5  # a small epsilon (small rand) is not beneficial to find global optimal
discountF = 0.9  # discount factor
targetUpdRate = 0.2

pBuffer = 0  # pointer
bufferSize = 2 ** 12
mBatchSize = 64
stateBuffer = np.zeros((stateNum,bufferSize),dtype = np.float64)
actionBuffer = np.zeros((bufferSize,),dtype = np.float64)
rewardBuffer = np.zeros((bufferSize,),dtype = np.float64)
nStateBuffer = copy.deepcopy(stateBuffer)
state = env.reset().reshape((stateNum,-1))
for pBuffer in range(bufferSize): # fill the buffer before the training starts
    stateBuffer[:,pBuffer] = state.reshape((stateNum,))
    action = policy(state,AdNN)
    actionBuffer[pBuffer] = action
    state, reward, _, _ = env.step(action)
    rewardBuffer[pBuffer],nStateBuffer[:,pBuffer] = reward,state.reshape((stateNum,))

qLabel = np.zeros((mBatchSize,))
for epoch in range(20):
    for pBuffer in range(bufferSize): # bufferSize
        stateBuffer[:,pBuffer] = state.reshape((stateNum,))
        action = policy(state,AdNN)
        actionBuffer[pBuffer] = action
        state, reward, _, _ = env.step(action)
        rewardBuffer[pBuffer],nStateBuffer[:,pBuffer] = reward,state.reshape((stateNum,))
        mBatchIdx = np.random.randint(0, bufferSize, (mBatchSize,))
        mStates,mNStates,mRewards,mActions = stateBuffer[:,mBatchIdx],nStateBuffer[:,mBatchIdx],rewardBuffer[mBatchIdx],actionBuffer[mBatchIdx]
        for i in range(mBatchSize): # mBatchSize
            inputA = mNStates[:,i].reshape((stateNum,-1))
            outputA = 2 * np.tanh(fp(inputA,AdNNTarget)[-1])
            outputA += np.random.normal(0,0.2,size = (1,1))
            if outputA[0,0] < -2.0:
                outputA[0,0] = -2.0
            elif outputA[0,0] > 2.0:
                outputA[0,0] = 2.0
            inputC = np.concatenate((inputA, outputA), axis = 0)
            tmp1 = fp(inputC, DNNTarget)[-1][0,0]
            tmp2 = fp(inputC, DNN2Target)[-1][0,0]
            qLabel[i] = mRewards[i]+discountF*np.minimum(tmp1,tmp2)
        # upd 2 critics
        inputs = np.concatenate((mStates,mActions.reshape(1,-1)),axis = 0)
        outList = fp(inputs,DNN)
        bp(inputs, outList, 2*(outList[-1] - qLabel.reshape((1,-1)))/layersDim[-1]/mBatchSize)
        outList = fp(inputs,DNN2)
        bp2(inputs, outList, 2*(outList[-1] - qLabel.reshape((1,-1)))/layersDim[-1]/mBatchSize)
        # upd actor
        if (pBuffer+1) % 2 == 0:
            outListA = fp(mStates,AdNN)
            outA = 2 * np.tanh(outListA[-1])
            outListC = fp(np.concatenate((mStates,outA),axis = 0),DNN)
            outC = outListC[-1]
            dOutA = bpNUPD(outListC,-1/mBatchSize*np.ones_like(outC))[-1,:].reshape((1,-1))
            bpAdNN(mStates,outListA,2 * dOutA * (1 - (outA/2)**2))
        # upd target DNNs
        for i in range(len(DNN)):
            DNNTarget[i] = (1 - targetUpdRate) * DNNTarget[i] + targetUpdRate * DNN[i]
        for i in range(len(DNN2)):
            DNN2Target[i] = (1 - targetUpdRate) * DNN2Target[i] + targetUpdRate * DNN2[i]
        for i in range(len(AdNN)):
            AdNNTarget[i] = (1 - targetUpdRate) * AdNNTarget[i] + targetUpdRate * AdNN[i]
    print(f'epoch:{epoch}--reward:{np.mean(rewardBuffer)}')
# test
for i in range(200):  # bufferSize
    action = tPolicy(state, AdNN)
    state, _, _, _ = env.step(action)
    env.render()
    time.sleep(0.0001)
env.close()
