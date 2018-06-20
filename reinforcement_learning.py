import numpy as np
from collections import OrderedDict
import sys

"""directions:
    ^ north = 1
    > east = 2
    v south = 3
    < west = 4"""

statesInOrder = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]


def buildEnvironment(r):
    envDict = OrderedDict()
    # Utility, []
    # envDict[state_num] = [utility, north[up,right,left], east[up,right,left], south[up,right,left], west[up,right,left], optimalPolicy]

    # First row
    envDict["s0"] = [r, ["s0", "s1", "s0"], ["s1", "s0", "s0"], ["s3", "s0", "s1"], ["s0", "s0", "s3"], 0]
    envDict["s1"] = [r, ["s1", "s2", "s0"], ["s2", "s1", "s1"], ["s1", "s0", "s2"], ["s0", "s1", "s1"], 0]
    envDict["s2"] = [r, ["s2", "t1", "s1"], ["t1", "s4", "s2"], ["s4", "s1", "t1"], ["s1", "s2", "s4"], 0]
    envDict["t1"] = [1]

    # Second Row
    envDict["s3"] = [r, ["s0", "s3", "s3"], ["s3", "s6", "s0"], ["s6", "s3", "s3"], ["s3", "s0", "s6"], 0]
    envDict["s4"] = [r, ["s2", "s5", "s4"], ["s5", "s8", "s2"], ["s8", "s4", "s5"], ["s4", "s2", "s8"], 0]
    envDict["s5"] = [r, ["t1", "s5", "s4"], ["s5", "s9", "t1"], ["s9", "s4", "s5"], ["s4", "t1", "s9"], 0]
    # Third Row
    envDict["s6"] = [r, ["s3", "s7", "s6"], ["s7", "s10", "s3"], ["s10", "s6", "s7"], ["s6", "s3", "s10"], 0]
    envDict["s7"] = [r, ["s7", "s8", "s6"], ["s8", "t2", "s7"], ["t2", "s6", "s8"], ["s6", "s7", "t2"], 0]
    envDict["s8"] = [r, ["s4", "s9", "s7"], ["s9", "t3", "s4"], ["t3", "s7", "s9"], ["s7", "s4", "t3"], 0]
    envDict["s9"] = [r, ["s5", "s9", "s8"], ["s9", "t4", "s5"], ["t4", "s8", "s9"], ["s8", "s5", "t4"], 0]

    # Fourth Row
    envDict["s10"] = [r, ["s6", "t2", "s10"], ["t2", "s10", "s6"], ["s10", "s10", "t2"], ["s10", "s6", "s10"], 0]
    envDict["t2"] = [1]
    envDict["t3"] = [-10]
    envDict["t4"] = [10]

    return envDict


#Print functions to format iteration prints
def printIteration(i, environment):
    print("it" + str(i), end=" ")
    for state in statesInOrder:
        if state.startswith("s"):
            print("{:7.3f}    ".format(environment[state][0]), end="")
    print("")


def printIterationQ(i, Q):
    print("it" + str(i), end=" ")
    for state in statesInOrder:
        for action in range(4):
            print("{:7.3f}    ".format(Q[state, action+1][1]), end="")
        print("|", end="")
    print("")


def printHeader(algorithm):
    print(algorithm)
    print("   ", end="")
    for state in statesInOrder:
        if state.startswith("s"):
            print("{:>7}    ".format(state), end="")
    print("")


def printHeaderQ():
    print("Q-learning")
    print("   ", end="")
    for state in statesInOrder:
        print("{:>30}               ".format(state), end="")
    print("")

#Getters for environment to improve code readability
def getUtility(environment, state):
    return environment[state][0]


def getNeighbors(environment, state, action):
    if state.startswith("s"):
        return environment[state][action]


def getPolicy(environment, state):
    return environment[state][-1]


def getPolicyArray(environment):

    policyArray=np.array([])
    for state in environment.keys():
        if state.startswith("s"):
            policyArray=np.append(policyArray,getPolicy(environment,state))
    return policyArray


def getUtilityArray(environment):

    utilityArray=np.array([])
    for state in environment.keys():
        if state.startswith("s"):
            utilityArray=np.append(utilityArray,getUtility(environment,state))
    return utilityArray


def getQValues(Q):
    qValues = np.zeros((16,4))
    for state,action in Q.keys():
        i = int(state[1:])
        j = action - 1
        qValues[i, j] = Q[state, action][1]
    return qValues


########################### Value Iteration ###########################

def utilityForOneDirection(environment, neighborStates, r = 0 , d=1 , p=(1, 0, 0)):

    sum = 0
    for i in range(3):
        n=neighborStates[i]
        sum += getUtility(environment, n) * p[i]
    return sum


def maxUtilityForOneState(environment,state, r=0, d=1, p=(1,0,0)):

    utilities = np.array([])
    for dir in range(4):
        currentUtility = utilityForOneDirection(environment, getNeighbors(environment, state, dir + 1), r, d, p)
        utilities = np.append(utilities, currentUtility)
    maxUtility = utilities.max()
    optimalPolicy = utilities.argmax() + 1  # Because of our notation

    return maxUtility, optimalPolicy


def maxUtilityForAllStates(environment, r=0, d=1, p=(1,0,0)):

    newUtilities = {}
    newPolicies = {}
    isConverged = True
    for state in environment.keys():
        if state.startswith("s"):
            maxUtility, optimalPolicy = maxUtilityForOneState(environment, state, r, d, p)
            maxUtility = d * maxUtility + r
            if maxUtility != environment[state][0]:
                isConverged = False
            newUtilities[state] = maxUtility
            newPolicies[state] = optimalPolicy

    for state in environment.keys():
        if state.startswith("s"):
            environment[state][0] = newUtilities[state]
            environment[state][-1] = newPolicies[state]
    return isConverged

def valueIteration(environment, r=0, d=1, p=(1,0,0)):
    counter = 1
    isConverged = False
    printHeader("Value Iteration")
    printIteration(0, environment)
    while isConverged == False:
        printIteration(counter, environment)
        isConverged = maxUtilityForAllStates(environment,r,d,p)
        counter += 1

    printIteration(counter, environment)
########################### End of value iteration ###########################


########################### Policy Iteration ###########################
def generatePolicy(environment):

    for state in environment.keys():
        if state.startswith('s'):
            environment[state][-1] = np.random.randint(0, 4) + 1


def util(environment, neighborStates, r = 0 , d=1 , p=(1, 0, 0)):

    sum = 0
    for i in range(3):
        n=neighborStates[i]
        sum += getUtility(environment, n) * p[i]

    return sum


def valueDetermination(environment, r = 0 , d=1 , p=(1, 0, 0)):
    for state in environment.keys():
        if state.startswith("s"):
            action=getPolicy(environment,state)
            n=getNeighbors(environment,state,action)
            environment[state][0] = d * util(environment, n, r, d, p) + r


def maxPolicy(environment, state, r, d, p):
    utilities = np.array([])
    for i in range(3):
        n = getNeighbors(environment, state, i+1)
        utilities = np.append(utilities, util(environment, n, r, d, p))

    maxUtil = utilities.max()
    bestPolicy = utilities.argmax()+1
    return maxUtil, bestPolicy


def policyIteration(environment, r, d, p=(1, 0, 0)):

    counter = 0
    generatePolicy(environment)
    printHeader("Policy Iteration")
    while(1):
        valueDetermination(environment, r, d, p)
        changed = False
        printIteration(counter, environment)
        for state in environment.keys():
            if state.startswith("s"):
                newUtil, newPolicy = maxPolicy(environment, state, r, d, p)
                policy = getPolicy(environment, state)
                n = getNeighbors(environment, state, policy)
                if newUtil > util(environment, n, r, d, p):
                    environment[state][-1] = newPolicy
                    changed = True
        counter += 1

        if changed == False:
            break
    printIteration(counter, environment)

########################### End of policy iteration ###########################


########################### Q-learning ###########################

def buildQ(environment):
    Q = OrderedDict()
    for state in environment.keys():
        if state.startswith("s"):
            Q[state, 1] = [environment[state][1][0], 0]
            Q[state, 2] = [environment[state][2][0], 0]
            Q[state, 3] = [environment[state][3][0], 0]
            Q[state, 4] = [environment[state][4][0], 0]

        else: #Terminal states
            Q[state, 1] = [state, 0]
            Q[state, 2] = [state, 0]
            Q[state, 3] = [state, 0]
            Q[state, 4] = [state, 0]

    return Q


def expectedQ(Q, environment, state, action, p):
    neighbors = getNeighbors(environment, state, action)
    sum = 0
    for i in range(3):
        neighbor = neighbors[i]
        _, QValue = maxQ(Q, neighbor)
        sum += QValue * p[i]
    return sum


def maxQ(Q, state):
    QValues = np.array([])
    for i in range(3):
        QValues = np.append(QValues, Q[state, i+1][1])

    return QValues.argmax()+1, QValues.max()


def updateQValue(Q, state, action, environment, a, d, p):
    nextState = getNeighbors(environment, state, action)[0]
    expectedQValue = expectedQ(Q, environment, state, action, p)
    environment[state][-1], _ = maxQ(Q, state)
    Q[state, action][1] += a * (getUtility(environment, nextState) + d * expectedQValue - Q[state, action][1])


def sumQValues(Q, state):
    sum = 0
    for i in range(4):
        sum += Q[state, i+1][1]
    return sum


def decideAction(Q, state, e):
    action = -1
	explore = np.random.uniform(0, 1)
    if(explore < e):
        action = np.random.randint(0, 4) + 1
    else:
        sum = sumQValues(Q, state)
        if sum > 0:
            action, _ = maxQ(Q, state)
        else:
            action = np.random.randint(0, 4) + 1

    return action


def Qlearning(environment, a, d, e, p, N):
    Q = buildQ(environment)
    startState = "s6"
    currentState = startState
    #printHeaderQ()
    for i in range(N):
     #   printIterationQ(i, Q)
        while currentState.startswith("s"):
            action = decideAction(Q, currentState, e)
            updateQValue(Q, currentState, action, environment, a, d, p)
            currentState = Q[currentState, action][0]
        currentState = startState

    return Q

########################### End of Q-learning ###########################

########################### Experiment functions ###########################
def VIexperiment(r, d, p):
    environment = buildEnvironment(r)
    valueIteration(environment, r, d, p)
    policies = getPolicyArray(environment)
    utilities = getUtilityArray(environment)

    return utilities, policies

def PIexperiment(r, d, p):
    environment = buildEnvironment(r)
    policyIteration(environment, r, d, p)
    policies = getPolicyArray(environment)
    utilities = getUtilityArray(environment)

    return utilities, policies

def QlearningExperiment(a, d, e, p, N):
    Q = Qlearning(environment, a, d, e, p, N)
    policies = getPolicyArray(environment)
    QValues = getQValues(Q)

    return QValues, policies


if __name__ == '__main__':
    np.random.seed(62)
    r = -0.01
    d = 0.9
    p = 0.8
    probability = (p, (1-p)/2, (1-p)/2)
    environment = buildEnvironment(r)
    VIutilities, VIpolicies = VIexperiment(r, d, probability)
    print(VIpolicies)
    PIutilities, PIpolicies = PIexperiment(r, d, probability)
    print(PIpolicies)
    e = 0.1
    a = 1
    N = 10000

    QValues, Qpolicies = QlearningExperiment(a, d, e, probability, N)
    print(QValues)
    print(Qpolicies)


