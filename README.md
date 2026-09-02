# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm
1.Initialize all Q values for every state and action to zero.
2.Set the starting exploration rate epsilon to one and choose decay and minimum values.
3.For each episode, reset the environment to get the initial state.
4.Generate one complete episode by following the current epsilon greedy policy derived from the Q table.
5.Record the total reward obtained in that episode and store it.
6.Traverse the episode backwards while accumulating the discounted return at each step.
7.Update the Q value of each visited state and action using the learning rate and the difference between the return and the current Q value.
8.After the episode, reduce epsilon by multiplying with the decay factor but never below the minimum.
9.After all episodes, derive the greedy policy by selecting the best action for each state from the Q table.
10.Estimate the state value function by taking the maximum Q value for each state.


## Python Program
```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
num_episodes = 30000
gamma = 0.95
alpha = 0.05

epsilon_start = 0.8
epsilon_min = 0.01
epsilon_decay = 0.9997

max_steps_per_episode = 150


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------
def generate_episode(epsilon):

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------
epsilon = epsilon_start

for episode_num in range(num_episodes):

    episode = generate_episode(epsilon)

    total_reward = sum(
        reward for _, _, reward in episode
    )

    episode_rewards.append(total_reward)

    # Calculate return
    G = 0

    for state, action, reward in reversed(episode):

        G = gamma * G + reward

        # Incremental Monte Carlo update
        Q[state, action] += alpha * (
            G - Q[state, action]
        )

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------
optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------
def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("Name:          ")
    print("Register Number:      ")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(optimal_policy)


success_rate = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    success_rate
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "Monte Carlo Control Learning Curve"
)

plt.grid(True)
plt.show()


env.close()

```


## Output

<img width="860" height="690" alt="image" src="https://github.com/user-attachments/assets/3dca23a3-66af-42df-aa59-82f384e47630" />

<img width="877" height="580" alt="image" src="https://github.com/user-attachments/assets/4ab4a231-3c68-4dec-bc6c-743cd450db82" />








## Result
```
The learned policy grid shows a clear path from the start to the goal while avoiding holes,
 and the estimated state‑value function has values close to 1 near the goal and along good paths,
with values near 0 for hole states and rarely visited states.
```


## Inference
```
The high average reward indicates that on‑policy Monte Carlo control with epsilon‑greedy exploration successfully learns
a near‑optimal policy for this task. States with high value are safe and lead reliably to the goal under the learned policy,
while zero‑value states correspond to holes or dead ends that the policy effectively avoids. Overall, the experiment confirms that
sample‑based Monte Carlo updates with decaying exploration are sufficient to solve this small deterministic control problem.


```

