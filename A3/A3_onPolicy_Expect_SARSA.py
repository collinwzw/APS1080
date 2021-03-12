import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
numberOfActions = env.action_space.n
# Num	Action
# 0	push left
# 1	no push
# 2	push right

# Reward
# -1 for each time step, until the goal position of 0.5 is reached.
# As with MountainCarContinuous v0, there is no penalty for climbing the left hill, which upon reached acts as a wall.

# Episode Termination
# The episode ends when you reach 0.5 position, or if 200 iterations are reached.
actionSpace = [0, 1, 2]
numberOfStates = 10 ** 2
GAMMA = 0.9
ALPHA = 0.05


def max_dict(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v


def create_bins():
    bins = np.zeros((2, 10))
    bins[0] = np.linspace(-1.20, 0.6, 10)
    bins[1] = np.linspace(-0.07, 0.07, 10)
    return bins


def assign_bins(observation, bins):
    state = np.zeros(2)
    for i in range(2):
        state[i] = np.digitize(observation[i], bins[i])
    return state


def get_state_as_string(state):
    string_state = ''.join(str(int(e)) for e in state)
    while len(string_state) != 2:
        string_state = '0' + string_state
    return string_state


def get_all_states_as_string():
    states = []
    for i in range(numberOfStates):
        states.append(str(i).zfill(2))
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
        pi[state] = np.random.choice(actionSpace)
    return pi


def initReturns():
    returns = {}

    all_states = get_all_states_as_string()
    for state in all_states:
        returns[state] = {}
        for action in range(env.action_space.n):
            returns[state][action] = []
    return returns


def play_onPolicy_Expected_SARSA(env,Q, bins,eps,display):
    obs = env.reset()
    state = get_state_as_string(assign_bins(obs, bins))
    # prev_screen = env.render(mode='rgb_array')
    # plt.imshow(prev_screen)
    count = 0
    complete = 0
    while True:
        count += 1
        if np.random.uniform() < eps:
            action = env.action_space.sample()  # epsilon greedy
        else:
            action = max_dict(Q[state])[0]

        obs, reward, done, info = env.step(action)

        if display == True:
            screen = env.render(mode='rgb_array')
            plt.imshow(screen)


        if done == True:
            if count < 200:
                reward = 1000
                complete = 1

        expected_q = 0
        new_state = get_state_as_string(assign_bins(obs, bins))
        max_action, max_q = max_dict(Q[new_state])
        greedy_actions = 0

        for i in actionSpace:
            if Q[new_state][i] == max_q:
                greedy_actions += 1

        non_greedy_action_probability = eps / len(actionSpace)
        greedy_action_probability = ((1 - eps) / greedy_actions) + non_greedy_action_probability

        for i in actionSpace:
            if Q[new_state][i] == max_q:
                expected_q += Q[new_state][i] * greedy_action_probability
            else:
                expected_q += Q[new_state][i] * non_greedy_action_probability


        Q[state][action] += ALPHA * (reward + GAMMA * expected_q - Q[state][action])
        state = new_state
        if done == True:
            break;
    return complete


def evaluate_and_plot_parameters(cumulative_completion, training_episodes):
    title = f'total episodes vs completed episodes'
    print(f'Evaluated {title}')
    line, = plt.plot(
        np.arange(1, training_episodes + 1),
        cumulative_completion,
        label=title)
    return line


if __name__ == '__main__':

    env = gym.make('MountainCar-v0')

    bins = create_bins()
    Q = initQ()

    onPolicy_SARSA_completeList = []
    completed = 0
    eps = 0.05
    mem = 0
    training_episodes = 5000
    for i in range(training_episodes):
        complete = play_onPolicy_Expected_SARSA(env, Q, bins, eps, False)
        if i % 100 == 0 and i != 0:
            print("i = " + str(i) + ", and completed over past 100 episode " + str(completed - mem))
            mem = completed

        completed += complete
        onPolicy_SARSA_completeList.append(completed)

    evaluate_and_plot_parameters(onPolicy_SARSA_completeList, training_episodes)
    plt.show()

    eps = 0
    avg = 0
    for i in range(100):
        complete = play_onPolicy_SARSA(env, Q, bins, eps, False)
        avg += complete
    print(avg / 100)
    # play(env, Q, bins,eps, True)



