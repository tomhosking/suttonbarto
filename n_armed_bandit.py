from random import randint
from math import sqrt

import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import numpy.ma.core as npy
import numpy as numpy
from numpy.random.mtrand import normal, uniform


def selectIndexWithProbability(probs):
    sum = 0
    idx=0
    rand = uniform()
    for p in probs:
        sum += p
        if rand < sum:
            return idx
        idx += 1
    return idx-1

numpy.seterr(all='ignore')

n = 20

stepmax = 600

samples = 40

epsilon = 0.90

temperature = .01

rewardresults =[0.] * stepmax
rightfractionresults = [0.] * stepmax

for steps in range(1,stepmax):
    averagereward=0
    rightcount=0
    runcount = 0
    for sample in range(0,samples):
        actionvalues = normal(0, 1, n)
        Qvalues = [0.] * n
        actioncount = [0] * n


        for i in range(0,steps):
            temperature = 1/sqrt(i+1)
            rand = uniform()
            if False and rand > epsilon:
                action = randint(0, n-1)
            else:
                 action = argmax(Qvalues)
                #energy = npy.exp(npy.array(Qvalues)/temperature)
                #probabilities = energy/sum(energy)
                #action = selectIndexWithProbability(probabilities)
                

            reward = actionvalues[action] + normal(0, 1)

            Qvalues[action] = (Qvalues[action]* actioncount[action] + reward)/ ( 1 + actioncount[action])

            actioncount[action] += 1

            averagereward += reward

            runcount += 1

        if argmax(Qvalues) == argmax(actionvalues):
            rightcount += 1

    rewardresults[steps] = averagereward/runcount
    rightfractionresults[steps] = rightcount/steps


plt.plot(rewardresults)
plt.show()
#plt.plot(rightfractionresults)
