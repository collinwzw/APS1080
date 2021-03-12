import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
numberOfActions = env.action_space.n
# Num	Action
# 0	push left
# 1	no push
# 2	push right

#Reward
# -1 for each time step, until the goal position of 0.5 is reached.
# As with MountainCarContinuous v0, there is no penalty for climbing the left hill, which upon reached acts as a wall.

# Episode Termination
# The episode ends when you reach 0.5 position, or if 200 iterations are reached.
actionSpace = [0,1,2]
numberOfStates = 10 ** 4
GAMMA = 0.9
ALPHA = 0.01


def max_dict(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v


def create_bins():

    bins = np.zeros((2, 100))
    bins[0] = np.linspace(-1.20, 0.6, 100)
    bins[1] = np.linspace(-0.07, 0.07, 100)
    return bins


def assign_bins(observation, bins):
    state = np.zeros(2)
    for i in range(2):
        state[i] = np.digitize(observation[i], bins[i])
    return state


def get_state_as_string(state):
    string_state = ''.join(str(int(e)) for e in state)
    return string_state


def get_all_states_as_string():
    states = []
    for i in range(numberOfStates):
        states.append(str(i).zfill(4))
    return states


def initQ():
    Q = {}

    all_states = get_all_states_as_string()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q

def initPolicy():
    pi = {}

    all_states = get_all_states_as_string()
    for state in all_states:
        pi[state]= np.random.choice(actionSpace)
    return pi

def initReturns():
    returns = {}

    all_states = get_all_states_as_string()
    for state in all_states:
        returns[state] = {}
        for action in range(env.action_space.n):
            returns[state][action] = []
    return returns

def play(env, policy,bins, display):
    obs = env.reset()
    # prev_screen = env.render(mode='rgb_array')
    # plt.imshow(prev_screen)
    rewards = []
    actions = []
    states = []
    totalRewards = 0
    while True:
        state = get_state_as_string(assign_bins(obs, bins))
        action = policy[state]
        actions.append(action)
        states.append(state)

        obs, reward, done, info = env.step(action)
        if display == True:
            screen = env.render(mode='rgb_array')
            plt.imshow(screen)
        totalRewards += reward
        rewards.append(reward)
        if done == True:
            if display == True:
                print(totalRewards)
            break

    return states, actions, rewards


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t - 100):(t + 1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t - 100):(t + 1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == '__main__':

    env = gym.make('MountainCar-v0')

    eps = 0.05
    gamma = 1

    bins = create_bins()
    Q = initQ()
    returns = initReturns()
    policy = initPolicy()
    rewardList = []
    avg = 0
    episode_rewards = []
    for i in range(2000):
        if i % 100 == 0 and i != 0:
            print("i = " + str(i) + ", and average reward over past 100 episode " + str(avg / 100))
            episode_rewards.append(avg)
            avg = 0
            states, actions, rewards = play(env, policy,bins, False)
        else:
            states, actions, rewards = play(env, policy,bins, False)

        seen = set()
        G = 0
        for i, (state, action, reward) in enumerate(zip(reversed(states), reversed(actions), reversed(rewards))):

            G = G * gamma + reward
            if (state,action) in seen:
                continue

            seen.add((state,action))
            returns[state][action].append(G)
            Q[state][action] = sum(returns[state][action]) / len(returns[state][action])
            maxAction = policy[state]
            if np.random.uniform() < 1 - eps:
                values = np.array([Q[state][a] for a in actionSpace])
                best = np.random.choice(np.where(values == values.max())[0])
                policy[state] = actionSpace[best]
            else:
                policy[state] = np.random.choice(actionSpace)
        avg += sum(rewards)
        rewardList.append(sum(rewards))


    plot_running_avg(episode_rewards)
    plt.show()
    eps = 0


    avg = 0

    for i in range(100):
        states, actions, rewards = play(env, policy,bins, False)
        avg += sum(rewards)
    print(avg/100)
    play(env, policy, bins, True)



