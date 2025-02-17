import math
import time
import torch
import numpy as np
from gymnasium.wrappers import FlattenObservation

class ReplayMemory():

    def __init__(self, state_dim, capacity, batch_size):
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminations = np.zeros(capacity, dtype=bool)
        self.capacity = capacity
        self.batch_size = batch_size
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, terminated):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.terminations[self.position] = terminated

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng):
        indices = rng.choice(self.size, self.batch_size, replace=False)
        return (
            self.states[indices], 
            self.actions[indices], 
            self.rewards[indices], 
            self.next_states[indices], 
            self.terminations[indices]
            )


class DQN(torch.nn.Module):

    def __init__(self, state_dim, n_actions, n_neurons):
        super(DQN, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_neurons, n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_neurons, n_actions))

    def forward(self, state):
       return self.network(state)


class DDQN():

    def __init__(
        self,
        env,
        is_constrained,
        num_episodes,
        milestone_freq,
        save_path,
        rep,
        offset,
        n_neurons,
        device,
        memory_size,
        learning_rate,
        batch_size,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        gamma,
        tau,
        policy_net_update_freq,
        target_net_update_freq,
        ):

        self.env = FlattenObservation(env)
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.is_constrained = is_constrained
        self.num_episodes = num_episodes
        self.milestone_freq = milestone_freq
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.rep = rep
        self.offset = offset
        
        torch.manual_seed(self.rep * 2e6 + offset)
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.policy_net = DQN(self.state_dim, self.n_actions, n_neurons).to(self.device)
        self.target_net = DQN(self.state_dim, self.n_actions, n_neurons).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(self.state_dim, memory_size, batch_size)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.policy_net_update_freq = policy_net_update_freq
        self.target_net_update_freq = target_net_update_freq

    def get_epsilon(self, episode):
        return max(self.epsilon_end, self.epsilon_start * self.epsilon_decay ** episode)

    def train(self):
        step = 0
        start = time.perf_counter()
        for episode in range(self.num_episodes):
            rng = np.random.default_rng(seed=int(self.rep * 1e6 + episode + 1e5 * self.offset))
            state, info = self.env.reset(seed=int(self.rep * 3e6 + episode + 1e5 * self.offset))
            terminated = False
            truncated = False
            while not(terminated or truncated):
               
                with torch.no_grad():
                    state_action_values = self.policy_net(
                        torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                        ).squeeze(0)
                    feasible_actions = np.arange(self.n_actions)

                    if self.is_constrained:
                        mask = torch.tensor(info['mask'], dtype=torch.bool, device=self.device)
                        state_action_values[~mask] = float('-inf')
                        feasible_actions = feasible_actions[info['mask']]
                    
                    epsilon = self.get_epsilon(episode)
                    if rng.random() < epsilon:
                        action = rng.choice(feasible_actions)
                    else:
                        action = state_action_values.argmax().item()

                    next_state, reward, terminated, truncated, info = self.env.step(action)

                    self.memory.add(state, action, reward, next_state, terminated)

                if self.memory.size >= self.memory.capacity // 10:
                    if step % self.policy_net_update_freq == 0:
                        self.update_policy_net(rng)
                    if step % self.target_net_update_freq == 0:
                        self.update_target_net()

                state = next_state
                step += 1

            if episode % self.milestone_freq == 0:
                end = time.perf_counter()
                elapsed_time = end - start
                start = end
                self.save(episode, step, elapsed_time)

        end = time.perf_counter()
        elapsed_time = end - start
        self.save(self.num_episodes, step, elapsed_time)

    def update_policy_net(self, rng):
        states, actions, rewards, next_states, terminations = self.memory.sample(rng)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)  # Indexing needs int64
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        terminations = torch.tensor(terminations, dtype=torch.bool, device=self.device)
        batch_indices = torch.arange(self.memory.batch_size, device=self.device)

        predictions = self.policy_net(states)[batch_indices, actions]

        with torch.no_grad():
            policy_net_next_state_max_actions = self.policy_net(next_states).argmax(dim=1)
            targets = rewards + self.gamma * self.target_net(next_states)[
                batch_indices, policy_net_next_state_max_actions
                ] * (1 - terminations.to(torch.float32))
        
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in target_net_state_dict.keys():
            target_net_state_dict[key] = self.tau * policy_net_state_dict[key] + \
                (1 - self.tau) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def save(self, episode, step, elapsed_time):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            # 'target_net_state_dict': self.target_net.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'memory': self.memory,
            'rep': self.rep,
            'episode': episode,
            'step': step,
            'elapsed_time': elapsed_time,
            }, self.save_path / f"rep{self.rep}_episode{episode}.pth")