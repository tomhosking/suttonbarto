__author__ = 'tom.hosking'

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import  poisson
from matplotlib import cm
from pylab import *
from numpy import ma
from time import  sleep

alpha = 0.5
epsilon = 0.1

gamma = 0.9

n = 10

numruns = 2000

policy = {}
Qvalue = {}

states = []
actions = []

finalstate = (n,n)
episodelengths = []

for x in range(-1,2):
    for y in range(-1,2):
        actions.append((x,y))

for x in range(0,n+1):
    for y in range(0,n+1):
        states.append((x,y))
        for a in actions:
            Qvalue[((x,y),a)] = 0
        policy[(x,y)] = (0,0)

def normpdf(x, mu=0, sigma=1):
    u = float((x-mu) / abs(sigma))
    y = np.exp(-u*u/2) / (np.sqrt(2*np.pi) * abs(sigma))
    return y

def GetTransition(s, a):

    #sprime = s
    #print(sprime[0])
    #sprime =(max(min(s[0] + a[0],n),0), max(min(s[1] + a[1],n),0))
    if (s[0] + a[0]) > n or (s[1] + a[1]) > n or (s[0] + a[0]) < 0 or (s[1] + a[1]) < 0:
        sprime = (s[0], s[1])
    else:
        sprime = (s[0]+a[0], s[1]+a[1])

    #if a is not (0,0):
    reward = -1
    dist = np.sqrt(pow(sprime[0]-float(n)/2, 2) + pow(sprime[1]-float(n)/2, 2))
    reward -= normpdf(dist, sigma = n/8) * 10
    if sprime == finalstate:
        reward += 5
    if (s[0] + a[0]) > n or (s[1] + a[1]) > n or (s[0] + a[0]) < 0 or (s[1] + a[1]) < 0:
        reward -= 50

    return (sprime,reward)

def GetAction(s):
    if np.random.random() > epsilon:
        return policy[s]
    else:
        #print("xplore")
        return actions[np.random.randint(0, len(actions))]

def UpdatePolicy(policy):
    for s in states:
        besta = 0
        bestret = False
        for a in actions:
            ret = Qvalue[(s,a)]
            if bestret is False or ret > bestret:
                besta = a
                bestret = ret
        policy[s] = besta
    return policy




policy = UpdatePolicy(policy)



X = np.arange(0, n+1, 1)
Y = np.arange(0, n+1, 1)
X, Y = np.meshgrid(X, Y)


U = np.zeros((n+1,n+1))
V = np.zeros((n+1,n+1))

#1

plt.ion()
fig = plt.figure()
Q = plt.quiver( U, V)
#qk = quiverkey(Q, 0.5, 0.92, 0, '', labelpos='W', fontproperties={'weight': 'bold'})

l,r,b,t = axis()
dx, dy = r-l, t-b
axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy])

title('Policy')

fig2 = plt.figure()

progr, = plot(range(0,len(episodelengths)), episodelengths)
ax2 = plt.axes()



for i in range(0,numruns):

    s = (0,0)
    a = GetAction(s)

    #print("Q after ",i,Qvalue)
    #print("p after ",i,policy)

    t=0

    terminated = False
    while not terminated:
        a = GetAction(s)
        (sprime,reward) = GetTransition(s,a)

        bestq = False
        for aprime in actions:
            thisq = Qvalue[sprime, aprime]
            if thisq > bestq or bestq is False:
                bestq = thisq
        Qvalue[(s,a)] = Qvalue[(s,a)] + alpha * (reward + gamma * bestq - Qvalue[(s,a)]) # Q learning
        # Qvalue[(s,a)] = Qvalue[(s,a)] + alpha * (reward + gamma * Qvalue[(sprime,aprime)] - Qvalue[(s,a)]) # sarsa
        s = sprime
        #a = aprime
        policy = UpdatePolicy(policy)

        if s == finalstate:
            terminated = True
            if i > 0:
                episodelengths.append(t)

        t +=1


    for s in states:
        U[s[0], s[1]] = policy[s][1]
        V[s[0], s[1]] = policy[s][0]
    Q.set_UVC(U, V)

    progr.set_data(range(0,len(episodelengths)), episodelengths)
    ax2.relim()
    ax2.autoscale_view()
    #fig.canvas.draw()
    #Q.draw()


    #if i==1:
    #    plt.show()

    #plt.draw()
    plt.xlabel("index" + str(i))
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    sleep(.1)




plt.show()
input("** Done **")