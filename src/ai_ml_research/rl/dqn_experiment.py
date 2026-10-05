import copy
import random
import numpy as np
import torch
from torch import optim
from torch.nn.functional import one_hot, smooth_l1_loss
from torchrl.envs import GymEnv
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from ai_ml_research.rl.agent import QFunction


def setup() -> tuple[int, torch.device]:
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.set_default_device(DEVICE)

    SEED = random.SystemRandom().randint(0, 2**32 - 1)
    torch.manual_seed(SEED)
    return SEED, DEVICE


SEED, DEVICE = setup()

print(f"SEED: {SEED}")

SAMPLE_BATCH_SIZE = 16
EPISODE_LENGTH = 4000
STORAGE_MEM_MAX = EPISODE_LENGTH
MIN_BUFFER_SIZE = SAMPLE_BATCH_SIZE * 3
TARGET_Q_FUNCTION_UPDATE_FREQ = 100
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 2000

env = GymEnv("CartPole-v1")

OBS_DIM = env.observation_spec["observation"].shape[-1]
ACTION_COUNT = env.action_spec.shape[-1]
REWARD_DIM = env.reward_spec.shape[-1]
HIDDEN_DIM = 128

print(f"Action count: {ACTION_COUNT}")
print(f"observation_spec: {env.observation_spec}")
print(f"action_spec: {env.action_spec}")
print(f"reward_spec: {env.reward_spec}")
print(f"done_spec: {env.done_spec}")


LEARNING_RATE = 0.01
GAMMA = 0.99

replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(STORAGE_MEM_MAX),
    # batch_size=SAMPLE_BATCH_SIZE,
)

q_function = QFunction(OBS_DIM, ACTION_COUNT, HIDDEN_DIM)
target_q_function = copy.deepcopy(q_function).requires_grad_(False)
loss_fn = smooth_l1_loss
optimiser = optim.Adam(params=q_function.parameters(), lr=LEARNING_RATE)


def get_epsilon_greedy_action(
    observation: torch.Tensor, epsilon: float, action_count: int
):
    observation_count = observation.ndim
    isSingleObservation = observation_count == 1

    if isSingleObservation:
        observation = observation.unsqueeze(0)

    # What actions are greedy or random?
    random_mask = torch.rand(observation_count) < epsilon
    greedy_mask = ~random_mask
    action_idx = torch.randint(0, action_count, (observation_count,))

    if greedy_mask.any():
        with torch.inference_mode():
            q_values = q_function(observation[greedy_mask])
        action_idx[greedy_mask] = q_values.argmax(dim=-1)

    action_one_hot = one_hot(action_idx, num_classes=action_count).to(
        torch.int64
    )

    if isSingleObservation:
        action_one_hot = action_one_hot.squeeze(0)

    return action_one_hot


td = env.reset()
global_step = 0

print("\n BEGIN EXPERIMENT \n")

for _ in range(EPISODE_LENGTH):
    print(f"Step: {global_step}")
    epsilon: float = EPS_END + (EPS_START - EPS_END) * np.exp(
        -global_step / EPS_DECAY
    )

    one_hot_action = get_epsilon_greedy_action(
        observation=td["observation"],
        epsilon=epsilon,
        action_count=ACTION_COUNT,
    )

    # one_hot_action = env.action_spec.rand()
    td["action"] = one_hot_action

    td = env.step(td)
    replay_buffer.add(td)

    # Optimise #
    if len(replay_buffer) > MIN_BUFFER_SIZE:
        batch = replay_buffer.sample(
            batch_size=min(len(replay_buffer), SAMPLE_BATCH_SIZE)
        ).to(DEVICE)

        q_values = q_function(batch["observation"])  # [batch, n_actions]

        # gather adds to the autograd graph - making sure backprop is through max action only.
        action_idx = batch["action"].argmax(dim=-1)
        current_q = q_values.gather(-1, action_idx.unsqueeze(1)).squeeze(1)
        with torch.inference_mode():
            next_q = (
                target_q_function(batch["next", "observation"])
                .max(dim=-1)
                .values
            )

        target_q = batch["next", "reward"].squeeze(-1) + (
            GAMMA * next_q * ~batch["next", "done"].squeeze(-1)
        )

        loss = smooth_l1_loss(input=current_q, target=target_q)
        if loss.item() is not None:
            print(f"{loss.item()}")

        optimiser.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(q_function.parameters(), max_norm=10)
        optimiser.step()

    # Reset when episode ends
    if td["next", "done"].item():
        print("Resetting MDP")
        td = env.reset()

    if global_step % TARGET_Q_FUNCTION_UPDATE_FREQ == 0:
        print("Update Target")
        target_q_function.load_state_dict(q_function.state_dict())

    global_step += 1

print(f"Data Sample Count: {len(replay_buffer)}")
print(f"Data Sample: {replay_buffer.sample(batch_size=1)["observation"]}")
# batch = replay_buffer.sample(batch_size=10)
