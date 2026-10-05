import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Hyperparameters
# -----------------------------
ENV_NAME = "CartPole-v1"

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64

BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE = 1_000

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10_000

TARGET_UPDATE_FREQ = 1_000
NUM_EPISODES = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Q-Network
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# Epsilon-greedy policy
# -----------------------------
def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(
        0
    )

    with torch.no_grad():
        q_values = policy_net(state)

    return q_values.argmax(dim=1).item()


# -----------------------------
# Training step
# -----------------------------
def train_step(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < MIN_BUFFER_SIZE:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(
        BATCH_SIZE
    )

    states = torch.tensor(states, device=DEVICE)
    actions = torch.tensor(actions, device=DEVICE)
    rewards = torch.tensor(rewards, device=DEVICE)
    next_states = torch.tensor(next_states, device=DEVICE)
    dones = torch.tensor(dones, device=DEVICE)

    # Q(s, a)
    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(s', a')
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]

        target_q = rewards + GAMMA * next_q * (1 - dones)

    # Huber loss
    loss = nn.functional.smooth_l1_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()

    # Optional gradient clipping
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)

    optimizer.step()

    return loss.item()


# -----------------------------
# Main training loop
# -----------------------------
def main():

    env = gym.make(ENV_NAME)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(DEVICE)

    target_net = DQN(state_dim, action_dim).to(DEVICE)

    # Initially identical
    target_net.load_state_dict(policy_net.state_dict())

    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    global_step = 0

    for episode in range(NUM_EPISODES):

        state, _ = env.reset()

        episode_reward = 0
        episode_loss = []

        done = False

        while not done:

            # Exponential epsilon decay
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(
                -global_step / EPS_DECAY
            )

            action = select_action(state, policy_net, epsilon, action_dim)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            loss = train_step(policy_net, target_net, optimizer, replay_buffer)

            if loss is not None:
                episode_loss.append(loss)

            # Update target network
            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            global_step += 1

        avg_loss = np.mean(episode_loss) if episode_loss else 0

        print(
            f"Episode {episode + 1:3d} | "
            f"Reward: {episode_reward:6.1f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Loss: {avg_loss:.4f}"
        )

    env.close()

    torch.save(policy_net.state_dict(), "dqn_cartpole.pth")


if __name__ == "__main__":
    main()
