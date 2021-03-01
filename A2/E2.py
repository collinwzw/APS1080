import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display


def getMaximumValueFromDictionary(dictionary):
    max_key = None
    max_value = float('-inf')
    for key, value in dictionary.items():
        if value > max_value:
            max_key = key
            max_value = value
    return max_key, max_value

def findNearestValue(a, b):
  index = np.digitize(x=a, bins=b)
  if index >= len(b):
      print(index)
      print(b)
  return b[index]


def getNeastState( obs, states):
  l_state = []
  for i in range(4):
    l_state.append(findNearestValue(obs[i], states[i]))
  return l_state  # returns the state values of the current observation variable

# Initialize policy
def initPi(combinations):
    pi = {}

    #initialize the random policy
    for state in combinations:
        key = state.tolist()
        pi[tuple(key)] = np.random.choice([0,1])
    return pi

def play(env, policy,v_ndarray,gamma):
    obs = env.env.state
    currentState = getNeastState(obs, v_ndarray)
    action = policy[tuple(currentState)]
    state_action_reward = [(currentState,action,1)]
    while True:

        obs, reward, done, info = env.step(action)
        currentState = getNeastState(obs, v_ndarray)
        if done == True:
            state_action_reward.append((currentState,None, 0))
            break
        else:
            action = policy[tuple(currentState)]
            state_action_reward.append((currentState, action, 1))
    G = 0
    state_action_return = []
    first = True

    for state, action, reward in reversed(state_action_reward):
        if first:
            first = False
        else:
            state_action_return.append((state,action,G))
        G = reward + gamma * G
    state_action_return.reverse()
    return state_action_return

env = gym.make("CartPole-v0")
env.reset()
gamma = 0.95

v_ndarray = [np.linspace(-2.6, 2.6, 10),  np.linspace(-500, 500, 10),
                       np.linspace(-0.5, 0.5, 10),  np.linspace(-500, 500, 10),]
mesh = np.array(np.meshgrid(v_ndarray[0], v_ndarray[1],v_ndarray[2],v_ndarray[3]))
combinations = mesh.T.reshape(-1, 4)
actionStateValueTable = {}
R = {}
for state in combinations:
  key =  state.tolist()
  actionStateValueTable[tuple(key)] = {}
  for action in range(0,2):
    actionStateValueTable[tuple(key)][action] = 0
    R[(tuple(key),action)] = []

policy = initPi(combinations)



for i in range(50000):
    #action = env.action_space.sample()
    state_action_return = play(env, policy,v_ndarray,gamma)
    print("episode " + str(i) )
    #obs [cart postion, cart velocity, pole angle, pole velocity at tip]

    #calculate Q
    seen = set()
    for state, action, G in reversed(state_action_return):
        stateActionPair = (tuple(state),action)
        if stateActionPair not in seen:
           R[stateActionPair].append(G)
           actionStateValueTable[tuple(state)][action] = np.mean(R[stateActionPair])
           seen.add(stateActionPair)

    #calculate new policy
    for state in actionStateValueTable.keys():
        action, _ = getMaximumValueFromDictionary(actionStateValueTable[state])
        policy[state] = action

