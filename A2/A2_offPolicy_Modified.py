import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
numberOfActions = env.action_space.n
actionSpace = [0,1]
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
    # obs[0] -> cart position --- -4.8 - 4.8
    # obs[1] -> cart velocity --- -inf - inf
    # obs[2] -> pole angle    --- -41.8 - 41.8
    # obs[3] -> pole velocity --- -inf - inf

    bins = np.zeros((4, 10))
    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(-5, 5, 10)
    bins[2] = np.linspace(-.418, .418, 10)
    bins[3] = np.linspace(-5, 5, 10)

    return bins


def assign_bins(observation, bins):
    state = np.zeros(4)
    for i in range(4):
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

def initC():
    C = {}

    all_states = get_all_states_as_string()
    for state in all_states:
        C[state] = {}
        for action in range(env.action_space.n):
            C[state][action] = 0
    return C

def play_one_game(bins, Q, eps=0.5):
    observation = env.reset()
    done = False
    cnt = 0  # number of moves in an episode
    state = get_state_as_string(assign_bins(observation, bins))
    total_reward = 0

    while not done:
        cnt += 1
        # np.random.randn() seems to yield a random action 50% of the time ?
        if np.random.uniform() < eps:
            act = env.action_space.sample()  # epsilon greedy
        else:
            act = max_dict(Q[state])[0]

        observation, reward, done, _ = env.step(act)

        total_reward += reward

        if done and cnt < 200:
            reward = -300

        state_new = get_state_as_string(assign_bins(observation, bins))

        a1, max_q_s1a1 = max_dict(Q[state_new])
        Q[state][act] += ALPHA * (reward + GAMMA * max_q_s1a1 - Q[state][act])
        state, act = state_new, a1

    return total_reward, cnt

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
        action = np.random.choice(policy[state])
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
        #print(totalRewards)
    return states, actions, rewards

def playTargetPolicy(env, policy,bins, display):
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
        #print(totalRewards)
    return states, actions, rewards

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t - 100):(t + 1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':

    env = gym.make('CartPole-v0')

    eps = 0.05
    gamma = 1

    bins = create_bins()
    Q = initQ()
    C = initC()

    targetPolicy = initPolicy()
    rewardList = []
    avg = 0
    episode_rewards = []
    for i in range(500):
        behaviorPolicy = {}
        for state in get_all_states_as_string():
            rand = np.random.uniform()
            if rand < 1 - eps:
                behaviorPolicy[state] = [targetPolicy[state]]
            else:
                behaviorPolicy[state] = actionSpace
        if i % 100 == 0 and i != 0:
            print("i = " + str(i) + ", and average reward over past 100 episode " + str(avg / 100))

            avg = 0
            states, actions, rewards = play(env, behaviorPolicy,bins, False)
        else:
            states, actions, rewards = play(env, behaviorPolicy,bins, False)

        G = 0
        W = 1
        for i, (state, action, reward) in enumerate(zip(reversed(states), reversed(actions), reversed(rewards))):

            G = G * gamma + reward

            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            values = np.array([Q[state][a] for a in actionSpace])
            best = np.random.choice(np.where(values == values.max())[0])
            targetPolicy[state] = actionSpace[best]
            if action != targetPolicy[state]:
                break;

            if len(behaviorPolicy[state]) == 1:
                prob = 1 - eps
            else:
                prob = eps / len(behaviorPolicy[state])
            W *= 1 / prob
        avg += sum(rewards)
        episode_rewards.append(sum(rewards))
    plot_running_avg(episode_rewards)
    plt.show()

    avg = 0

    for i in range(100):
        states, actions, rewards = playTargetPolicy(env, targetPolicy,bins, False)
        avg += sum(rewards)
    print(avg/100)
    playTargetPolicy(env, targetPolicy, bins, True)



