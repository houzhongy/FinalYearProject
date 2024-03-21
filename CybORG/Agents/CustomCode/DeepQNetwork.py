import gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import collections
from collections import namedtuple

class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, X):
        return self.model(X)
    
class QNetwork:
    def __init__(self, input_dims, output_dims, lr, logdir=None):
        self.net = FullyConnectedModel(input_dims, output_dims)
        self.lr = lr 
        self.logdir = logdir
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
    
class ReplayMemory:
    def __init__(self, memory_size=50000):
        # Initializes the replay memory, which stores transitions recorded from the agent taking actions in the environment.
        self.memory_size = memory_size
        self.memory = collections.deque([], maxlen=memory_size)

    def sample_batch(self, batch_size=64):
        # Returns a batch of randomly sampled transitions to be used for training the model.
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        # Appends a transition to the replay memory.
        self.memory.append(transition)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN_Agent:

    def __init__(self, env, input_dims, output_dims,action_space, lr=5e-4, fe=None, render=False):
        # Initialize the DQN Agent.
        self.env = env
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.action_space = action_space
        self.lr = lr
        self.policy_net = QNetwork(self.input_dims, self.output_dims, self.lr)
        self.target_net = QNetwork(self.input_dims, self.output_dims, self.lr)
        self.target_net.net.load_state_dict(self.policy_net.net.state_dict())  # Copy the weight of the policy network
        self.rm = ReplayMemory()
        self.batch_size = 64
        self.gamma = 0.99
        self.c = 0

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        # Implement an epsilon-greedy policy. 
        p = random.random()
        if p > epsilon:
            with T.no_grad():
                return self.greedy_policy(q_values)
        else:
            return T.tensor([[self.action_space.sample()]], dtype=T.long)

    def greedy_policy(self, q_values):
        # Implement a greedy policy for test time.
        return T.argmax(q_values)
    
    def get_action(self, state):
        with T.no_grad():
            q_values = self.policy_net.net(state)

        # Decide the next action with epsilon greedy strategy
        action = self.epsilon_greedy_policy(q_values).reshape(1, 1)

        return action
    
    def store_transition(self, transition):
        self.rm.memory.append(transition)

    def train(self):
            if len(self.rm.memory) < 64:
                return
            # Sample minibatch with size N from memory
            transitions = self.rm.sample_batch(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = T.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=T.bool)
            non_final_next_states = T.cat([s for s in batch.next_state if s is not None])
            state_batch = T.cat(batch.state)
            action_batch = T.cat(batch.action)
            reward_batch = T.cat(batch.reward)

            # Get current and next state values
            state_action_values = self.policy_net.net(state_batch).gather(1, action_batch) # extract values corresponding to the actions Q(S_t, A_t)
            next_state_values = T.zeros(self.batch_size)
            
            with T.no_grad():
                next_state_values[non_final_mask] = self.target_net.net(non_final_next_states).max(1)[0] # extract max value
                
            # Update the model
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            criterion = T.nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            self.policy_net.optimizer.step()

            # Update the target Q-network in each 50 steps
            self.c += 1
            if self.c % 5 == 0:
                self.target_net.net.load_state_dict(self.policy_net.net.state_dict())

    def test(self, modified_state=0,  feature_extractor=None):
        # Evaluates the performance of the agent every 20 episodes.
        MAX_STEPS_PER_EPISODE = 50
        state = self.env.reset()
        rewards = []

        for t in range(MAX_STEPS_PER_EPISODE):
            if modified_state:
                state = feature_extractor(T.from_numpy(state))
            else:
                state = T.from_numpy(state)
            with T.no_grad():
                q_values = self.policy_net.net(state)
            action = self.greedy_policy(q_values)
            try:
                state, reward, terminated, _ = self.env.step(action.item())
            except:
                reward = -0.1
                state = state.numpy()
            finally:
                rewards.append(reward)
                if terminated:
                    break

        return np.sum(rewards)
