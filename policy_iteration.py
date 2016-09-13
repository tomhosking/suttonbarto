
import grid
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import  poisson
from matplotlib import cm
## Initialisation

mu_add = (3,2)
mu_remove = (3,4)

policy = {}
statevalue = {}

movecost = -2
payment = 10

gamma = 0.9

n=15

def GetChangeProb(delta):
    px = 0;
    py = 0;
    for i in range(-20,20):
        px += poisson.pmf(i, mu_add[0]) * poisson.pmf(-delta[0] + i, mu_remove[0])
        py += poisson.pmf(i, mu_add[1]) * poisson.pmf(-delta[1] + i, mu_remove[1])
        #print("px ", delta, " add ", i, " rem ", -delta[0] + i, poisson.pmf(i, mu_add[0]) * poisson.pmf(-delta[0] + i, mu_remove[0]), " tot ", px)
        #print("py ", delta, " add ", i, " rem ", -delta[1] + i, poisson.pmf(i, mu_add[1]) * poisson.pmf(-delta[1] + i, mu_remove[1]), " tot ", py)
    #print("TOT ", px*py)
    #if delta == (7,-1):
    #    quit()
    return px*py

def GetTransitionProb(s, sprime, action):
    deltax = sprime[0] - s[0] + action
    deltay = sprime[1] - s[1] - action
    return GetChangeProb((deltax,deltay))

def GetReward(s, sprime, action):
    reward = abs(action) * movecost
    reward -= min(0, payment * (sprime[0] - s[0] + action)) # add reward if cars went out
    reward -= min(0, payment * (sprime[1] - s[1] - action))
    #print("Reward for ", s, " -> ", sprime, ", action ", action, " is ", reward, " p ", GetTransitionProb(s, sprime, action))
    return reward


states = []
for x in range(0,n+1):
    for y in range(0,n+1):
        states.append((x,y))

## Init
for s in states:
    policy[s] = 0
    statevalue[s] = 0

error = 0
errorthresh = 1e-1
runs=0

stable = True


while (stable == False and runs < 10) or runs == 0  :
    stable = True
    ## Policy eval
    print("Policy eval #", runs)
    evalruns=0
    while evalruns == 0 or error > errorthresh:

        error = 0
        for s in states:
            temp = 0#statevalue[s]

            #print("Evaluating for s=", s)
            for sprime in states:
                temp += GetTransitionProb(s, sprime, policy[s]) * (GetReward(s, sprime, policy[s]) + gamma * statevalue[sprime])

            print("Value of ", s, " is ", temp, ", err ", max(error, abs(temp - statevalue[s])), ", errold ", error, " olds ", statevalue[s])
            error = max(error, abs(temp - statevalue[s]))
            statevalue[s] = temp
        evalruns += 1
        print(error, " > ", errorthresh, "?")



    ## Policy improvement
    print("Policy imp #", runs)
    for s in states:
        temp = policy[s]
        # find argmax_a of the Q value
        print("Improving for s=", s)
        bestreturn = False
        besta = 0
        for a in range(-4,5):
            ret = 0
            for sprime in states:
                ret += GetTransitionProb(s, sprime, a) * (GetReward(s, sprime, a) + gamma * statevalue[sprime])
            if bestreturn == False or ret > bestreturn:
                bestreturn = ret
                besta = a
        policy[s] = besta
        if besta != temp:
            stable = False

    print("Best policy after ", runs, " is: ", policy)
    runs +=1



print("Value: ", statevalue)
print(stable)
print(runs)

X = np.arange(0, n+1, 1)
Y = np.arange(0, n+1, 1)
X, Y = np.meshgrid(X, Y)

Zpolicy = np.zeros((n+1,n+1))
Zvalue = np.zeros((n+1,n+1))

for s in states:
    Zpolicy[s[0], s[1]] = policy[s]
    Zvalue[s[0], s[1]] = statevalue[s]

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Zpolicy, rstride=1, cstride=1, cmap=cm.RdYlBu, antialiased=True)
fig.canvas.set_window_title('Policy')


fig2 = plt.figure(figsize=plt.figaspect(0.5))
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(X, Y, Zvalue, rstride=1, cstride=1, cmap=cm.RdYlBu, antialiased=True)
fig2.canvas.set_window_title('State value')

plt.show()