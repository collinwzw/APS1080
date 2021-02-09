import gym
import numpy as np

gym.envs.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.74
)
# Create the gridworld-like environment
env=gym.make('FrozenLakeNotSlippery-v0')
# Let's look at the model of the environment (i.e., P):
#env.env.P


# Question: what is the data in this structure saying? Relate this to the course
# presentation of P

#Initialize state value function
def initV(env):
    V = np.zeros(env.env.ncol * env.env.nrow)
    return V


# Initialize policy
def initPi(env):
    pi = {}
    #initialize the random policy
    for state in env.env.P.keys():
        for action in env.env.P.get(state).keys():
            if state in pi.keys():
                pi[state].append(action)
            else:
                pi[state] = [action]
    return pi


# policy evaluation function
def policyEvaluation(env, V, pi, theta, gamma):
    episode = 0
    exit = False
    while not exit:
        delta = 0
        episode += 1
        #print(episode)

        for state in env.env.P.keys():
            oldV = V[state]
            sum = 0
            for action in pi[state]:
                new_state = env.env.P.get(state).get(action)[0][1]
                p = env.env.P.get(state).get(action)[0][0]
                reward = env.env.P.get(state).get(action)[0][2]
                policyWeight = 1 / len(pi[state])
                sum += policyWeight * p * (reward + gamma * V[new_state])

            V[state] = sum
            delta = max(delta, np.absolute(oldV - V[state]))
        if delta < theta:
            exit = True
    return V, episode

# policy Improvement function
def policyImprovement(env, V, pi, gamma):

    stable = True
    newpi = {}
    for state in env.env.P.keys():
        oldAction = pi[state]
        newAction = []
        newValue = []

        for action in env.env.P.get(state).keys():
            new_state = env.env.P.get(state).get(action)[0][1]
            p = env.env.P.get(state).get(action)[0][0]
            reward = env.env.P.get(state).get(action)[0][2]
            newValue.append(p * (reward + gamma * V[new_state]))
            newAction.append(action)

        #find maximum value index in newValue list
        newValue = np.array(newValue)
        bestValue = np.where(newValue == newValue.max())[0]
        bestActions = [newAction[item] for item in bestValue]
        newpi[state] = bestActions

        if bestActions != oldAction:
            stable = False


    return stable,newpi

# Value Iteration function
def valueIteration(env, V, pi, theta, gamma):
    episode = 0
    exit = False
    while not exit:
        delta = 0
        episode += 1
        #print(episode)

        for state in env.env.P.keys():
            oldV = V[state]
            sum = 0
            newValue = []
            for action in pi[state]:
                new_state = env.env.P.get(state).get(action)[0][1]
                p = env.env.P.get(state).get(action)[0][0]
                reward = env.env.P.get(state).get(action)[0][2]
                newValue.append(p * (reward + gamma * V[new_state]))

            newValue = np.array(newValue)
            bestV = np.where(newValue == newValue.max())[0]
            bestState = np.random.choice(bestV)
            V[state] = newValue[bestState]
            delta = max(delta, np.absolute(oldV - V[state]))
        if delta < theta:
            exit = True

    for state in env.env.P.keys():
        newAction = []
        newValue = []

        for action in env.env.P.get(state).keys():
            new_state = env.env.P.get(state).get(action)[0][1]
            p = env.env.P.get(state).get(action)[0][0]
            reward = env.env.P.get(state).get(action)[0][2]
            newValue.append(p * (reward + gamma * V[new_state]))
            newAction.append(action)

        #find maximum value index in newValue list
        newValue = np.array(newValue)
        bestValue = np.where(newValue == newValue.max())[0]
        bestActions = newAction[bestValue[0]]
        pi[state] = bestActions
    return V, episode, pi



V = initV(env)
# print(V)
pi = initPi(env)
gamma = 0.95
theta = 1e-6

# V_eval, episode = policyEvaluation(env, V, pi, theta, gamma)
# V_eval = np.reshape(V_eval,(4,4))
# print(episode)
# print(V)

#Policy Iteration
# V = initV(env)
# pi = initPi(env)
# stable = False
# while not stable:
#     V, episode = policyEvaluation(env, V, pi, theta, gamma)
#     stable,pi = policyImprovement(env, V, pi, gamma)
#     print(pi)
# print(pi)

V = initV(env)
pi = initPi(env)
V, episode, newpi = valueIteration(env, V, pi, theta, gamma)
print(episode)
print(V)
print(newpi)