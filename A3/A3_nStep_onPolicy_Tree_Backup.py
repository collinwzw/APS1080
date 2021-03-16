import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools
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
GAMMA = 1
ALPHA = 0.05
eps = 0.05

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


def evaluate_and_plot_parameters(cumulative_completion, training_episodes):
    title = f'total episodes vs completed episodes'
    print(f'Evaluated {title}')
    line, = plt.plot(
        np.arange(1, training_episodes + 1),
        cumulative_completion,
        label=title)
    return line


def play_nStep_onPolicy_Tree_Backup(env, Q, bins, eps, n, display):
    obs = env.reset()
    state = get_state_as_string(assign_bins(obs, bins))
    if np.random.uniform() < eps:
        action = env.action_space.sample()  # epsilon greedy
    else:
        action = max_dict(Q[state])[0]
    # prev_screen = env.render(mode='rgb_array')
    # plt.imshow(prev_screen)
    count = 0
    complete = 0
    T = np.inf

    states = [state]
    actions = [action]
    rewards = [0]

    for t in itertools.count():

        if t < T:
            obs, reward, done, info = env.step(action)
            count += 1
            if display == True:
                screen = env.render(mode='rgb_array')
                plt.imshow(screen)
            state = get_state_as_string(assign_bins(obs, bins))
            states.append(state)
            rewards.append(reward)
            if done:
                T = t + 1

                if count < 199:
                    complete = 1
                    # print('episode ends at step', t)
            else:
                if np.random.uniform() < eps:
                    action = env.action_space.sample()  # epsilon greedy
                else:
                    action = max_dict(Q[state])[0]
                actions.append(action)

        tau = t - n + 1

        if tau >= 0:
            G = 0
            if t + 1 >= T:
                G = rewards[T]
            else:
                expected_q = 0
                max_action, max_q = max_dict(Q[state])
                greedy_actions = 0
                for i in actionSpace:
                    if Q[state][i] == max_q:
                        greedy_actions += 1

                non_greedy_action_probability = eps / len(actionSpace)
                greedy_action_probability = ((1 - eps) / greedy_actions) + non_greedy_action_probability
                for i in actionSpace:
                    if Q[state][i] == max_q:
                        expected_q += Q[state][i] * greedy_action_probability
                    else:
                        expected_q += Q[state][i] * non_greedy_action_probability
                G = rewards[t + 1] + GAMMA * expected_q

            for k in range(min(t, T - 1), tau + 1, -1):

                state_k = states[k]
                expected_q = 0
                max_action, max_q = max_dict(Q[state_k])
                greedy_actions = 0
                for i in actionSpace:
                    if Q[state_k][i] == max_q:
                        greedy_actions += 1

                non_greedy_action_probability = eps / len(actionSpace)
                greedy_action_probability = ((1 - eps) / greedy_actions) + non_greedy_action_probability
                for i in actionSpace:
                    if Q[state_k][i] == max_q:
                        expected_q += Q[state_k][i] * greedy_action_probability
                    else:
                        expected_q += Q[state_k][i] * non_greedy_action_probability

                if actions[k] == max_action:
                    G = rewards[k] + GAMMA * expected_q + GAMMA * greedy_action_probability * G
                else:
                    G = rewards[k] + GAMMA * expected_q + GAMMA * non_greedy_action_probability * G

            state_action = (states[tau], actions[tau])
            Q[state_action[0]][state_action[1]] += ALPHA * (G - Q[state_action[0]][state_action[1]])
        # print('tau ', tau, '| Q %.2f' %  Q[states[tau]][actions[tau]], actions[tau])
        if tau == T - 1:
            break

    return complete, np.sum(rewards)


if __name__ == '__main__':
    debug = True
    env = gym.make('MountainCar-v0')

    bins = create_bins()
    Q = initQ()

    nStep_onPolicy_Tree_Backup_completeList = []
    rewards = 0
    completed = 0
    training_episodes = 5000
    mem = 0
    for i in range(training_episodes):
        complete, reward = play_nStep_onPolicy_Tree_Backup(env, Q, bins, eps, 2, False)
        rewards += reward
        completed += complete
        if i % 100 == 0 and i != 0 and debug:
            print("i = " + str(i) + ", and completed over past 100 episode " + str(rewards / 100))
            print("i = " + str(i) + ", and completed over past 100 episode " + str(completed - mem))
            mem = completed
            rewards = 0

        nStep_onPolicy_Tree_Backup_completeList.append(completed)
