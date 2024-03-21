import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, n_actions, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax()
        )
    
    def forward(self, X):
        return self.model(X)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, X):
        return self.model(X)

class A2C:
    def __init__(self,env, input_dims, output_dims):
        self.gamma = 0.99
        self.lr_actor = 3e-4
        self.lr_critic = 5e-4

        self.env = env
        self.action_dim = output_dims
        self.state_dim = input_dims

        self.actor = Actor(self.action_dim,self.state_dim)
        self.critic = Critic(self.state_dim)

        self.actor_optim = T.optim.Adam(self.actor.parameters(),lr=self.lr_actor)
        self.critic_optim = T.optim.Adam(self.critic.parameters(),lr=self.lr_critic)

        self.loss = nn.MSELoss()

    def get_action(self,state):
        actions = self.actor(state)
        dist = Categorical(actions)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(),log_prob

    def train(self,log_prob,state,state_,rew,done):
        gamma = 0.99
        v = self.critic(state)
        v_ = self.critic(state_)

        advantage = rew + (1-done)*gamma*v_ - v
        critic_loss = advantage.pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        loss_actor = -log_prob * advantage.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

    def test(self, modified_state=0, feature_extractor=None):
        # Evaluates the performance of the agent over 20 episodes.
        MAX_STEPS_PER_EPISODE = 50
        state = self.env.reset()
        state = T.tensor(state)
        rews = []

        for step in range(MAX_STEPS_PER_EPISODE):
            with T.no_grad():     
                if modified_state:
                    state = feature_extractor(state)
                action,log_prob = self.get_action(state)
                try:
                    state, reward, terminated, _ = self.env.step(action)
                except:
                    reward = -0.1
                    state = state
                state = T.tensor(state)
                rews.append(reward)
            if terminated:
                break

        return np.sum(rews)