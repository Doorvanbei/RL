import numpy as np
'''
title: MAGE (env: my pendulum-v0)
author: doorvanbei
date: 20210525
'''
def relu(x):
    return np.maximum(x, 0)
def reluder(x):
    return np.float64(x > 0)
def tan2(x):
    return 2.0*np.tanh(x)
def tan2der(y):
    return 2.0 - y ** 2 /2.0
def stateTrans(s,a): # dim(s) = (2,b), dim(a) = (1,b)
    x,y = s[0],s[1]
    Y = y + 0.15 * (a[0] + 5 * np.sin(x))
    X = x + 0.05 * Y
    Y = np.clip(Y,-8,8)
    X = ( X + np.pi ) % (2 * np.pi) - np.pi
    return np.array([X,Y])
def getReward(s,a):
    r = -(s[0] ** 2 + 0.1 * s[1] ** 2 + 0.001 * a[0] ** 2)
    return np.array([r])


np.random.seed(1)
gamma = .95
dense1units = 64
lRateCdNN = .01
lRateAdNN = .005

# create actor and critic DNN and their target networks
aw1 = np.random.random((dense1units,2))-0.5
ab1 = 0.00*np.random.random((dense1units,1))
aw2 = np.random.random((dense1units,dense1units))-0.5
ab2 = 0.00*np.random.random((dense1units,1))
aw3 = np.random.random((1,dense1units))-0.5
ab3 = 0.00*np.random.random((1,1))
taw1 = np.copy(aw1)
tab1 = np.copy(ab1)
taw2 = np.copy(aw2)
tab2 = np.copy(ab2)
taw3 = np.copy(aw3)
tab3 = np.copy(ab3)
cw1 = np.random.random((dense1units,3))-0.5
cb1 = 0.00*np.random.random((dense1units,1))
cw2 = np.random.random((dense1units,dense1units))-0.5
cb2 = 0.00*np.random.random((dense1units,1))
cw3 = np.random.random((1,dense1units))-0.5
cb3 = 0.00*np.random.random((1,1))
tcw1 = np.copy(cw1)
tcb1 = np.copy(cb1)
tcw2 = np.copy(cw2)
tcb2 = np.copy(cb2)
tcw3 = np.copy(cw3)
tcb3 = np.copy(cb3)

state = np.array([[0],[0]]) # initial state
bufferSize = 128
mBatchSize = 8
stateBuffer = np.zeros((2,bufferSize))

for pBuffer in range(bufferSize): # initial buffer
    stateBuffer[:,pBuffer] = state.reshape((-1,))
    if np.random.random() > 0.7:
        action = np.array([[np.random.uniform(-2,2)]])
    else:
        x1 = relu(aw1 @ state + ab1)
        x2 = relu(aw2 @ x1 + ab2)
        action = tan2(aw3 @ x2 + ab3)
    state = stateTrans(state, action)

for epoch in range(55):
    for pBuffer in range(bufferSize):
        stateBuffer[:,pBuffer] = state.reshape((-1,))
        mBatchIdx = np.random.randint(0, bufferSize, (mBatchSize,))
        state = stateBuffer[:,mBatchIdx]
        # forward function
        x1 = relu(aw1 @ state + ab1)
        x2 = relu(aw2 @ x1 + ab2)
        action = tan2(aw3 @ x2 + ab3)

        reward = -(state[0] ** 2 + 0.1 * state[1] ** 2 + 0.001 * action[0] ** 2).reshape((1,-1))
        x,y,a = state[0],state[1],action[0]
        Y = y + 0.15 * (a + 5 * np.sin(x))
        Y = np.clip(Y,-8,8)
        X = x + 0.05 * (y + 0.15 * (a + 5 * np.sin(x)))
        X = ( X + np.pi ) % (2 * np.pi) - np.pi
        nState = np.array([X,Y])

        xt1 = relu(taw1 @ nState + tab1)
        xt2 = relu(taw2 @ xt1 + tab2)
        nAction = tan2(taw3 @ xt2 + tab3)

        inputC = np.concatenate((state,action))
        y1 = relu(cw1 @ inputC + cb1)
        y2 = relu(cw2 @ y1 + cb2)
        q = cw3 @ y2 + cb3

        inputTc = np.concatenate((nState,nAction))
        ty1 = relu(tcw1 @ inputTc + tcb1)
        ty2 = relu(tcw2 @ ty1 + tcb2)
        nq = tcw3 @ ty2 + tcb3


        # begin to do dphi
        # part I
        dnqdinputTc = tcw1.T @ ( tcw2.T @ ( tcw3.T * reluder(ty2) ) * reluder(ty1) )
        dnqdnState = taw1.T @ (taw2.T @ (taw3.T @ (dnqdinputTc[-1].reshape((1,-1)) * tan2der(nAction)) * reluder(xt2)) * reluder(xt1)) + dnqdinputTc[0:-1].reshape((2,-1))
        clipEight = np.ones((1,mBatchSize)) * (nState[-1] < 8) * (nState[-1] > -8)
        dnqda = 0.05 * 0.15 * dnqdnState[0] + 0.15 * clipEight * dnqdnState[1]
        dqda = (cw1[:,-1].reshape((-1,1))).T @ (cw2.T @ (cw3.T * reluder(y2)) * reluder(y1))
        drda = -0.002 * action
        ddeltada = gamma * dnqda + drda - dqda # a sample per column
        den = np.linalg.norm(ddeltada)
        d_dqda_d_cw1 = np.zeros((mBatchSize,dense1units, 3))
        d_dqda_d_cb1 = np.zeros((mBatchSize,dense1units, 1))
        d_dqda_d_cw2 = np.zeros((mBatchSize,dense1units, dense1units))
        d_dqda_d_cb2 = np.zeros((mBatchSize,dense1units, 1))
        d_dqda_d_cw3 = np.zeros((mBatchSize,1, dense1units))
        d_dqda_d_cb3 = np.zeros((mBatchSize,1, 1))
        for sample in range(mBatchSize):
            inputCs = inputC[:,sample].reshape((-1,1))
            y1s = relu(cw1 @ inputCs + cb1)
            y2s = relu(cw2 @ y1s + cb2)
            qs = cw3 @ y2s + cb3
            d_dqda_d_cw1[sample][:,-1] = (cw2.T @ (cw3.T * reluder(y2s)) * reluder(y1s)).reshape((-1,))
            d_dqda_d_cw2[sample] = (cw1[:, -1].reshape((-1, 1)) * reluder(y1s) @ (cw3.T * reluder(y2s)).T).T
            d_dqda_d_cw3[sample] = (cw2 @ ( cw1[:, -1].reshape((-1, 1)) * reluder(y1s) ) * reluder(y2s)).T
        dcw1,dcw2,dcw3,dcb1,dcb2,dcb3 = 0,0,0,0,0,0
        for sample in range(mBatchSize):
            dcw1 -= ddeltada[0,sample] * d_dqda_d_cw1[sample]
            dcw2 -= ddeltada[0,sample] * d_dqda_d_cw2[sample]
            dcw3 -= ddeltada[0,sample] * d_dqda_d_cw3[sample]
            dcb1 -= ddeltada[0,sample] * d_dqda_d_cb1[sample]
            dcb2 -= ddeltada[0,sample] * d_dqda_d_cb2[sample]
            dcb3 -= ddeltada[0,sample] * d_dqda_d_cb3[sample]
        dcw1 /= den
        dcw2 /= den
        dcw3 /= den
        dcb1 /= den
        dcb2 /= den
        dcb3 /= den
        # part II
        bAid = np.ones((1,mBatchSize))
        delta = gamma * nq + reward - q
        den = np.linalg.norm(delta)
        dcb3 += 0.2 * (-2.0/den)*delta @ bAid.T
        dcw3 += 0.2 * (-2.0/den)*delta @ y2.T
        dcb2 += 0.2 * (cw3.T @ ((-2.0/den)*delta) * reluder(y2)) @ bAid.T
        dcw2 += 0.2 * (cw3.T @ ((-2.0/den)*delta) * reluder(y2)) @ y1.T
        dcb1 += 0.2 * (cw2.T @ (cw3.T @ ((-2.0/den)*delta) * reluder(y2)) * reluder(y1)) @ bAid.T
        dcw1 += 0.2 * (cw2.T @ (cw3.T @ ((-2.0/den)*delta) * reluder(y2)) * reluder(y1)) @ inputC.T
        # upd phi
        cw1 -= lRateCdNN * dcw1
        cw2 -= lRateCdNN * dcw2
        cw3 -= lRateCdNN * dcw3
        cb1 -= lRateCdNN * dcb1
        cb2 -= lRateCdNN * dcb2
        cb3 -= lRateCdNN * dcb3
        if (pBuffer+1) % 2 == 0:
            dbefAction = (cw1[:,-1].reshape((-1,1))).T @ (cw2.T @ (cw3.T @ (1/mBatchSize * np.ones((1,mBatchSize))) * reluder(y2)) * reluder(y1)) * tan2der(action)
            dab3 = dbefAction @ bAid.T
            daw3 = dbefAction @ x2.T
            dab2 = (aw3.T @ dbefAction * reluder(x2)) @ bAid.T
            daw2 = (aw3.T @ dbefAction * reluder(x2)) @ x1.T
            dab1 = (aw2.T @ (aw3.T @ dbefAction * reluder(x2)) * reluder(x1)) @ bAid.T
            daw1 = (aw2.T @ (aw3.T @ dbefAction * reluder(x2)) * reluder(x1)) @ state.T
            # upd theta
            aw1 += lRateAdNN * daw1
            aw2 += lRateAdNN * daw2
            aw3 += lRateAdNN * daw3
            ab1 += lRateAdNN * dab1
            ab2 += lRateAdNN * dab2
            ab3 += lRateAdNN * dab3

        if (pBuffer+1) % 4 == 0:
            taw1 = 0.9 * taw1 + 0.1 * aw1
            taw2 = 0.9 * taw2 + 0.1 * aw2
            taw3 = 0.9 * taw3 + 0.1 * aw3
            tab1 = 0.9 * tab1 + 0.1 * ab1
            tab2 = 0.9 * tab2 + 0.1 * ab2
            tab3 = 0.9 * tab3 + 0.1 * ab3
            tcw1 = 0.9 * tcw1 + 0.1 * cw1
            tcw2 = 0.9 * tcw2 + 0.1 * cw2
            tcw3 = 0.9 * tcw3 + 0.1 * cw3
            tcb1 = 0.9 * tcb1 + 0.1 * cb1
            tcb2 = 0.9 * tcb2 + 0.1 * cb2
            tcb3 = 0.9 * tcb3 + 0.1 * cb3

        state = stateBuffer[:,pBuffer].reshape((-1,1))
        if np.random.random() > 0.7:
            action = np.array([[np.random.uniform(-2, 2)]])
        else:
            x1 = relu(aw1 @ state + ab1)
            x2 = relu(aw2 @ x1 + ab2)
            action = tan2(aw3 @ x2 + ab3)
        state = stateTrans(state, action)
    R = 0
    for i in range(200):
        x1 = relu(aw1 @ state + ab1)
        x2 = relu(aw2 @ x1 + ab2)
        action = tan2(aw3 @ x2 + ab3)
        R += getReward(state,action)
        state = stateTrans(state, action)
    print(f'avg R:{R/200}')



