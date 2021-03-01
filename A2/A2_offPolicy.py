#offpolicy
import numpy as np
import gym
import matplotlib.pyplot as plt
from functools import partial

def discretize_val(val, min_val, max_val, num_states):
    """
	Discretizes a single float
	if val < min_val, it gets a discrete value of 0
	if val >= max_val, it gets a discrete value of num_states-1

	Args:
	    val (float): value to discretize
	    min_val (float): lower bound of discretization
	    max_val (float): upper bound of discretization
	    num_states (int): number of discrete states

	Returns:
	    float: discrete value
	"""
    state = int(num_states * (val - min_val) / (max_val - min_val))
    if state >= num_states:
        state = num_states - 1
    if state < 0:
        state = 0
    return state

def observationToState(num_states, lower_bounds, upper_bounds, obs):
    """
	Turns an observation into a discrete state

	Args:
	    num_states (list): list of number of states for each dimension of observation
	    lower_bounds (list): list of lowerbounds for discretization
	    upper_bounds (list): list of upperbounds for discretization
	    obs (list): observation in R^N to discretize

	Returns:
	    int: discrete state
	"""
    state_idx = []
    for ob, lower, upper, num in zip(obs, lower_bounds, upper_bounds, num_states):
        state_idx.append(discretize_val(ob, lower, upper, num))

    return np.ravel_multi_index(state_idx, num_states)

def GreedyPolicy(Q, state):

    max_val = Q[state, :].max()
    max_indices = np.where(np.abs(Q[state, :] - max_val) < 1e-5)[0]
    rand_idx = np.random.randint(len(max_indices))
    max_action = max_indices[rand_idx]

    return max_action



def EpsilonGreedyPolicy(Q, eps, state):
    sample = np.random.random_sample()

    num_actions = Q.shape[1]

    if sample > eps:
        max_val = Q[state, :].max()
        max_indices = np.where(np.abs(Q[state, :] - max_val) < 1e-5)[0]
        rand_idx = np.random.randint(len(max_indices))
        max_action = max_indices[rand_idx]

        return max_action
    else:
        return np.random.randint(num_actions)

# Initialize policy
def initPi(Q, eps):
    pi = partial(GreedyPolicy, Q, eps)
    return pi

# Initialize policy
def initB(Q, eps):
    b = partial(EpsilonGreedyPolicy, Q, eps)
    return b

def updatePi(Q):
    pi = partial(GreedyPolicy, Q)
    return pi

def updateB(Q, eps):
    b = partial(EpsilonGreedyPolicy, Q, eps)
    return b

def initQ(stateLength, numberOfActions):
    Q = np.zeros((stateLength, numberOfActions))
    return Q

def initC(stateLength, numberOfActions):
    C = np.zeros((stateLength, numberOfActions))
    return C

def initReturns(stateLength, numberOfActions):
    returns = {}
    for i, value in enumerate(Q):
        returns[i,0] = []
        returns[i,1] = []
    return returns

def play(env, policy,stateMapping, display):
    obs = env.reset()
    # prev_screen = env.render(mode='rgb_array')
    # plt.imshow(prev_screen)
    rewards = []
    actions = []
    states = []
    totalRewards = 0
    while True:
        state = stateMapping(obs)
        action = policy(state)
        actions.append(action)
        states.append(state)

        obs, reward, done, info = env.step(action)
        if display == True:
            screen = env.render(mode='rgb_array')
            plt.imshow(screen)
        totalRewards += reward
        rewards.append(reward)
        if done == True:
            #print(obs)
            break


        #print(totalRewards)
    return states, actions, rewards

env = gym.make('CartPole-v1')

eps = 0.05
gamma = 1
numberOfActions = env.action_space.n
numberOfStates = [10, 8, 10, 8]
lowerBounds = [-2.4, -9999, -0.21, -9999]
upperBounds = [2.4, 9999, 0.21, 9999]
stateMapping = partial(observationToState, numberOfStates, lowerBounds, upperBounds)
stateLength = np.prod(np.array(numberOfStates))
Q = initQ(stateLength, numberOfActions)
C = initC(stateLength, numberOfActions)
targetPolicy = initPi(Q, eps)
rewardList = []
avg = 0
behaviorPolicy = initB(Q, eps)
for i in range(20000):
    behaviorPolicy = updateB(Q,eps)
    if i%100 == 0 and i != 0:
        print("i = " + str(i) + ", and average reward over past 100 episode " + str(avg/ 100))
        avg = 0
        states, actions, rewards = play(env, behaviorPolicy, stateMapping, False)
    else:
        states, actions, rewards = play(env, behaviorPolicy,stateMapping,False)

    G = 0
    W = 1
    for i, (state, action, reward) in enumerate(zip(reversed(states), reversed(actions), reversed(rewards))):
        if i == 0:
            G = G * gamma
        else:
            G = G * gamma + rewards[i - 1]

        C[state, action] += W
        Q[state, action] += (W/C[state, action]) * (G - Q[state, action])
        targetPolicy = updatePi(Q)
        if action != targetPolicy(state) :
            break;

        if targetPolicy(state) == action:
            # selected greedy action
            prob = 1 - eps
        else:
            prob = eps/numberOfActions
        W *= 1/prob

    avg += sum(rewards)
    rewardList.append(sum(rewards))
plt.plot(rewardList)
plt.show()


policy = updatePi(Q)
avg = 0
for i in range(100):
    states, actions, rewards = play(env, policy, stateMapping, False)
    avg += sum(rewards)
print(avg/100)



