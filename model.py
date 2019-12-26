import torch
import torch.nn as nn

def init_weights(m):
    classname = m.__class__.__name__
    if 'conv' in classname.lower() or 'linear' in classname.lower():       
        nn.init.xavier_uniform_(m.weight.data)

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, n_actors, categorical=False):
        super().__init__()

        self.n_actors = n_actors
        self.observation_space = observation_space
        self.action_space = action_space
        self.categorical = categorical
        
        self.body = nn.Sequential(
                nn.Linear(self.observation_space, self.observation_space),
                nn.ReLU(),
                nn.Linear(self.observation_space, self.observation_space*2),
                nn.ReLU()
            ).apply(init_weights)

        self.actor = nn.Sequential(
                nn.Linear(self.observation_space*2, self.observation_space),
                nn.ReLU(),
                nn.Linear(self.observation_space, self.action_space)
            ).apply(init_weights)
        
        self.critic = nn.Sequential(
                nn.Linear(self.observation_space*2, self.observation_space),
                nn.ReLU(),
                nn.Linear(self.observation_space, 1)
            ).apply(init_weights)
        
        self.normal_std = nn.Parameter(torch.zeros(1, self.action_space))
        self.dist = torch.distributions.Normal
        
    def forward(self, x):
        x = x.reshape(-1, self.observation_space)
        
        x = self.body(x)
        v = self.critic(x)
        m = self.actor(x)
        
        s = self.normal_std.expand_as(m).exp()
        dist = self.dist(m, s)       
        
        return dist, v