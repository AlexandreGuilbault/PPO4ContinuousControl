import torch
import torch.nn as nn

import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

from model import ActorCritic


class RandomAgent():
    def __init__(self, observation_space, action_space, n_actors):
        
        self.observations_space = observation_space
        self.action_space = action_space
        self.n_actors = n_actors
    
    def act(self, obs, deploy=True):
        actions = np.random.randn(self.n_actors, self.action_space)
        actions = np.clip(actions, -1, 1)
        return actions
    
    def get_num_actors(self):
        return self.n_actors
    

class PPOAgent():
    def __init__(self, observation_space, action_space, n_actors, memory_size, lr=1e-4, 
                 eps=1e-5, batch_size=256, epoch=4, gamma=0.99, clip=0.2, max_grad=0.2,
                 value_coef=0.5, entropy_coef=0.01, seed=0):

        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available() : self.device='cuda'
        else : self.device = 'cpu'
        
        
        self.model = ActorCritic(observation_space,
                                 action_space,
                                 n_actors).to(self.device)  
        
        self.short_term_memory = ShortTermMemory(action_space=action_space, 
                                                 observation_space=observation_space,
                                                 n_actors=n_actors,
                                                 device=self.device,
                                                 buffer_size=memory_size,
                                                 gamma=gamma)
        
        self.optimizer = optimizer = optim.Adam(self.model.parameters(), 
                                                lr=lr, 
                                                eps=eps)      
        
        self.batch_size = batch_size
        self.epoch = epoch
        
        self.clip = clip
        self.max_grad = max_grad
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
    def act(self, obs, deploy=False):

        obs = torch.from_numpy(obs).float().to(self.device)
        
        dist, value = self.model(obs)
        action = dist.sample()
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        if deploy:
            return action.detach().cpu().numpy()
        else:
            return action, value, action_log_probs

    def memorize(self, observations, actions, action_log_probs, value_preds, rewards):       
        self.short_term_memory.add(observations, actions, action_log_probs, value_preds, rewards)    

    def hindsight(self):
        self.short_term_memory.compute_returns()
        
    def recall(self, batch_size):
        return self.short_term_memory.sample(batch_size)
    
    def learn(self):
        
        self.hindsight() # Calculate returns and advantages
        
        for epoch_i in range(self.epoch):
            samples = self.recall(self.batch_size)

            value_losses = []
            action_losses = []
            entropy_losses = []
            losses = []
            for obs, actions, returns, old_action_log_probs, advantages in samples:                       
                values, action_log_probs, dist_entropy = self.evaluate_actions(obs, actions)

                ratio = torch.exp(action_log_probs.squeeze() - old_action_log_probs.squeeze())

                surr1 = ratio * advantages.squeeze()
                surr2 = torch.clamp(ratio, 1.-self.clip, 1.+self.clip) * advantages.squeeze()

                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((returns.squeeze() - values.squeeze())**2).mean()
                loss = (action_loss + self.value_coef*value_loss - self.entropy_coef*dist_entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
                self.optimizer.step()

                value_losses.append(value_loss.item())
                action_losses.append(action_loss.item())
                entropy_losses.append(dist_entropy.item())
                losses.append(loss.item())
        
        self.reset_memory() # Prepare memory for next run

        return np.mean(losses), np.mean(action_losses), np.mean(value_losses), np.mean(entropy_losses)
    
    def set_train(self, train=True):
        self.model.train(train)
    
    def evaluate_actions(self, obs, actions):
        dist, value = self.model(obs)        
        
        action_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action_log_probs, dist_entropy
    
    def reset_memory(self):
        self.short_term_memory.reset()
        
    def get_num_steps(self):
        return self.short_term_memory.episode_steps

    def get_num_actors(self):
        return self.short_term_memory.n_actors    

    def memory_size(self):
        return len(self.short_term_memory)
    
    def save(self, folder, file):
        torch.save(self.model.state_dict(), folder + file)
        
    def load(self, folder, file):
        self.model.load_state_dict(torch.load(folder + file))
        self.set_train(False)
        
        
        
class ShortTermMemory:
    def __init__(self, action_space, observation_space, n_actors=1, device='cpu', buffer_size=1000, gamma=0.99):

        self.device = device
        
        self.buffer_size = buffer_size
        self.gamma = gamma
                
        self.n_actors = n_actors
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.observations = torch.zeros((self.buffer_size, self.n_actors, self.observation_space)).to(self.device)
        self.actions = torch.zeros((self.buffer_size, self.n_actors, self.action_space)).to(self.device)
        self.action_log_probs = torch.zeros((self.buffer_size, self.n_actors, 1)).to(self.device)
        self.values = torch.zeros((self.buffer_size, self.n_actors, 1)).to(self.device)
        self.rewards = torch.zeros((self.buffer_size, self.n_actors, 1)).to(self.device)        

        self.reset()
        
    def reset(self):        
        self.episode_steps = 0
        
    def add(self, observations, actions, action_log_probs, values, rewards):
        
        observations = torch.from_numpy(observations).float().to(self.device)
        
        i = self.episode_steps%self.buffer_size
        
        self.observations[i] = observations.reshape(self.observations.shape[1:])
        self.actions[i] = actions.reshape(self.actions.shape[1:])
        self.action_log_probs[i] = action_log_probs.reshape(self.action_log_probs.shape[1:])
        self.values[i] = values.reshape(self.values.shape[1:])
        self.rewards[i] = rewards.reshape(self.rewards.shape[1:])
        
        self.episode_steps += 1

    def compute_returns(self):
        # The code below migh seem complex but it is only a 
        # tensor computation of the following code slightly modified from :
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py
        #
        #         returns = torch.zeros(self.episode_steps, self.n_actors, 1).to(self.device)
        #         returns[-1] = self.rewards[self.episode_steps-1]
        #         for step in reversed(range(self.episode_steps-1)):
        #             returns[step] = returns[step + 1] * self.gamma + self.rewards[step]
        #         returns = returns[:self.episode_steps]
        #
        # This code greatly helped me to debug and better understand the algorithm. Thanks Ilya Kostrikov!

        with torch.no_grad():
            s,a = self.episode_steps, self.n_actors
            discounts = self.gamma**torch.arange(s, dtype=torch.float, device=self.device).unsqueeze(1)
            discount_adjustment = (1/self.gamma)**torch.arange(s, dtype=torch.float, device=self.device)
            discounts = (discounts.expand(s,s)).tril() * discount_adjustment.expand(s,s)
            self.returns = (self.rewards.permute(1,0,2).repeat(1,1,s) * discounts.repeat(a,1,1)).sum(1)
            self.returns = self.returns.permute(1,0).unsqueeze(-1)

            self.advantages = self.returns - self.values[:s]
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-10)
            
    def sample(self, batch_size):
    
        s = self.episode_steps
        a = self.n_actors
        indices = np.random.choice(s*a, size=((s*a)//batch_size, batch_size), replace=False)

        for i in indices:

            batch_obs = self.observations[:s].reshape(-1,self.observation_space)[[i]] 
            batch_action_log_prob = self.action_log_probs[:s].reshape(-1, 1)[[i]]
            batch_actions = self.actions[:s].reshape(-1, self.action_space)[[i]]
            
            batch_returns = self.returns[:s].reshape(-1, 1)[[i]]
            batch_advantages = self.advantages[:s].reshape(-1, 1)[[i]]
  
            yield batch_obs, batch_actions, batch_returns, batch_action_log_prob, batch_advantages

    def __len__(self):
        return self.episode_steps