import numpy as np
import copy
import gym
'''
pendulum: using SAC
author: doorvanbei
date: 20210411
'''
def relu(x):
    return np.maximum(x, 0)
def reluder(y):
    return np.float64(y > 0)
def softPlus(x):
    return np.log(1.0+np.e ** x)
def softPlusder(y):
    return 1.0 - np.e ** (-y)
def tan2(x):
    return 2.0*np.tanh(x)
def tan2der(y):
    return 2.0 - y ** 2 /2.0
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
def bp(inputs,DNN,outList,dOut,lRate): # backward propagation
    dx,dDNN = copy.deepcopy(dOut),copy.deepcopy(DNN)
    bAid = np.ones((1,inputs.shape[1]))
    for i in range(1, int(len(DNN)/2)):
        dDNN[-2 * i + 1] = dx @ bAid.T
        dDNN[-2 * i] = dx @ outList[-i - 1].T
        dx = (DNN[-2 * i].T @ dx) * reluder(outList[-i - 1])
    dDNN[1] = dx @ bAid.T
    dDNN[0] = dx @ inputs.T
    for i in range(len(DNN)):
        DNN[i] -= lRate * dDNN[i]
    return DNN
def bpNUPD(DNN,outList,dOut): # backward propagation
    dx = copy.deepcopy(dOut)
    for i in range(1, int(len(DNN)/2)):
        dx = (DNN[-2 * i].T @ dx) * reluder(outList[-i - 1])
    return DNN[0].T @ dx
def yGauss(x,mu,D):
    return (2.0*np.pi*D)**(-0.5) * np.e ** (-(x-mu)**2.0 / 2.0 / D)
def xyGaussGen(mu,D):
    x = np.random.normal(mu,D)
    return x, yGauss(x,mu,D) # return the action chosen, and its probability density
def actionClip(x0):
    x = copy.deepcopy(x0)
    for i,e in enumerate(x):
        if e > 2.0:
            x[i] = 2.0
        elif e < -2.0:
            x[i] = -2.0
    return x
stateNum = 3
np.random.seed(1)
env = gym.make('Pendulum-v0')
env.seed(1)
aLayersDim,cLayersDim = [stateNum,64,2],[4,64,1]
aDNN,cDNN1,cDNN2 = buildDNN(aLayersDim),buildDNN(cLayersDim),buildDNN(cLayersDim)
tDNN1,tDNN2 = copy.deepcopy(cDNN1),copy.deepcopy(cDNN2)
alpha = 1.0
discountF = 0.9
pBuffer = 0
bufferSize = 2 ** 10
mBatchSize = 64
lRateCritic = 0.001
lRateActor = 0.1
lRateAlpha = 0.001
targetEntropy = 0
targetUpdRate = 0.1
stateBuffer = np.zeros((stateNum,bufferSize),dtype = np.float64)
actionBuffer = np.zeros((4,bufferSize),dtype = np.float64) # store mu, D, x, y
rewardBuffer = np.zeros((bufferSize,),dtype = np.float64)
nStateBuffer = copy.deepcopy(stateBuffer)
# fill the buffer before training starts
state = env.reset().reshape((stateNum,-1))
for pBuffer in range(bufferSize):
    stateBuffer[:,pBuffer] = state.reshape((-1,))
    out = fp(state, aDNN)[-1]
    out[0],out[1] = tan2(out[0]),softPlus(out[1]) # out[0] -- tan2 --> mu, out[1] -- softPlus --> D
    x,y = xyGaussGen(out[0,0],out[1,0])
    if x > 2.0:
        x = 2.0
    if x < -2.0:
        x = -2.0
    actionBuffer[0,pBuffer],actionBuffer[1,pBuffer],actionBuffer[2,pBuffer],actionBuffer[3,pBuffer] = out[0,0],out[1,0],x,y # mu, D, action, possibility density
    state, reward, _, _ = env.step(np.array([[x]],dtype = np.float64))
    nStateBuffer[:,pBuffer] = state.reshape((-1,))
    rewardBuffer[pBuffer] = reward
# training
for episode in range(4):
    for pBuffer in range(bufferSize): # bufferSize
        stateBuffer[:,pBuffer] = state.reshape((-1,))
        out = fp(state, aDNN)[-1]
        out[0],out[1] = tan2(out[0]),softPlus(out[1])
        x,y = xyGaussGen(out[0,0],out[1,0])
        if x > 2.0:
            x = 2.0
        if x < -2.0:
            x = -2.0
        actionBuffer[0,pBuffer],actionBuffer[1,pBuffer],actionBuffer[2,pBuffer],actionBuffer[3,pBuffer] = out[0,0],out[1,0],x,y
        state, reward, _, _ = env.step(np.array([[x]]))
        nStateBuffer[:,pBuffer] = state.reshape((-1,))
        rewardBuffer[pBuffer] = reward
        mBatchIdx = np.random.randint(0, bufferSize-1, (mBatchSize,))
        mStates,mNStates,mRewards,mActions,mNActions = stateBuffer[:,mBatchIdx],nStateBuffer[:,mBatchIdx],rewardBuffer[mBatchIdx],actionBuffer[:,mBatchIdx],actionBuffer[:,mBatchIdx+1]
        # make q Label
        inputC = np.concatenate((mNStates,mNActions[2].reshape((1,-1))),axis = 0)
        outC = np.minimum(fp(inputC,tDNN1)[-1],fp(inputC,tDNN2)[-1])
        H = 1.0/2.0*np.log(2.0 * np.pi * np.e * mNActions[1])
        qLabel = mRewards + discountF * outC + alpha * H
        # critic upd
        inputC = np.concatenate((mStates,mActions[2].reshape(1,-1)),axis = 0)
        outCList1 = fp(inputC,cDNN1)
        outC1 = outCList1[-1]
        outCList2 = fp(inputC,cDNN2)
        outC2 = outCList2[-1]
        cDNN1 = bp(inputC,cDNN1,outCList1,2.0/mBatchSize*(outC1-qLabel),lRateCritic)
        cDNN2 = bp(inputC,cDNN2,outCList2,2.0/mBatchSize*(outC2-qLabel),lRateCritic)
        # actor upd
        if pBuffer % 2 == 1:
            outList = fp(mStates,aDNN)
            out = outList[-1]
            out[0], out[1] = tan2(out[0]), softPlus(out[1]) # mu, D
            rv = np.random.normal(0.0,1.0,(1,mBatchSize))
            a = rv * (out[1] ** 0.5) + out[0]
            a = a.reshape((-1,))
            aCut = actionClip(a)
            inputC = np.concatenate((mStates,aCut.reshape((1,-1))),axis = 0)
            outCList1 = fp(inputC,cDNN1)
            outC1 = outCList1[-1]
            dA1 = bpNUPD(cDNN1,outCList1,-1.0/mBatchSize*np.ones_like(outC1))[-1]
            outCList2 = fp(inputC,cDNN2)
            outC2 = outCList2[-1]
            dA2 = bpNUPD(cDNN2,outCList2,-1.0/mBatchSize*np.ones_like(outC2))[-1]
            for i,l in enumerate(dA2): # dAction = dA1
                if a[i] < -2.0 or a[i] > 2.0:
                    dA1[i] = 0.0
                    continue
                if outC1[0,i] > outC2[0,i]:
                    dA1[i] = l
            dD = rv * 0.5 * (out[1] ** (-0.5)) * dA1 - alpha*0.5/mBatchSize/out[1]
            dMuBefAct = dA1 * tan2der(out[0]) # dA1 = dMu
            dDBefAct = dD * softPlusder(out[1])
            dOut = np.concatenate((dMuBefAct.reshape((1,-1)), dDBefAct), axis = 0)
            aDNN = bp(mStates, aDNN, outList, dOut, lRateActor)
        # update alpha
        if pBuffer % 4 == 1:
            out = softPlus(fp(mStates,aDNN)[-1])
            H = 0.5*np.log(2.0 * np.pi * np.e * out[1])
            con = targetEntropy - H
            con **= 2
            con = np.mean(con)
            alpha -= lRateAlpha * 2*con*alpha
        # soft upd target critics
        if pBuffer % 2 == 1:
            for i in range(len(tDNN1)):
                tDNN1[i] = (1 - targetUpdRate) * tDNN1[i] + targetUpdRate * cDNN1[i]
            for i in range(len(tDNN2)):
                tDNN2[i] = (1 - targetUpdRate) * tDNN2[i] + targetUpdRate * cDNN2[i]
    print(f'in episode {episode}, reward {np.mean(rewardBuffer)}')
# check play
for i in range(400):
    out = fp(state, aDNN)[-1]
    out[0], out[1] = tan2(out[0]), softPlus(out[1])
    x, y = xyGaussGen(out[0, 0], out[1, 0])
    if x > 2.0:
        x = 2.0
    if x < -2.0:
        x = -2.0
    state, _, _, _ = env.step(np.array([[x]]))
    env.render()
env.close()

# these 2 functions are not used in this program
# def dydD(D,y): # d pDensity / d variance
#     return -D*y*(0.5 + np.log((2*np.pi*D)**(0.5) * y))
# def dydmu(mu,D,x,y): # d pDensity / d mean
#     return y / D * (x - mu)