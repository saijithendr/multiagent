import numpy as np
import random
import copy
from collections import namedtuple, deque


import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size,  random_seed):        
        self.state_size = state_size
        self.action_size = action_size        
        self.seed = random.seed(random_seed)
        self.batch_size = BATCH_SIZE   
        self.t_step = 0
        self.seed = random.seed(random_seed)
        self.decay_step = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.memory =  ReplayBuffer(BATCH_SIZE, BUFFER_SIZE, random_seed)
        
        # update target networks
        #self.soft_update(self.critic_local, self.critic_target, 1.0)
        #self.soft_update(self.actor_local, self.actor_target, 1.0)
        
        # Noise process
        #self.noise = OUNoise()
        self.noise = OUNoise( action_size , random_seed)        
      
    def step(self, state, action, reward, next_state, done,agent_number, beta):
        """Save experience in replay memory, and use random sample from buffer to learn."""        
        # Save experience / reward                                       
        self.memory.add(state, action, reward, next_state, done)            
       
        # Learn, if enough samples are available in memory  
        # Learn every UPDATE_EVERY time steps.
        self.t_step +=1
        
        if self.t_step %UPDATE_EVERY == 0:
            if self.memory.is_filled():                
                self.learn(agent_number, GAMMA,beta) 
    
   
     
    
    def numpy_to_torch(self,data):
        return torch.from_numpy(data).float().to(device)
                                
    def act(self, state, episode_num, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 👈 Add unsqueeze(0) to make it 2D
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().squeeze(0)  # 👈 squeeze back to 1D
        self.actor_local.train()
        if add_noise:
            action = self.add_random_noise(action, episode_num)
        return np.clip(action, -1, 1)


    def noise_decay_schedule(self,episode_num):  
        return max(0.0, NOISE_SCALE * (1 - (episode_num / NOISE_DECAY_LIMIT)))

    def add_random_noise(self,action,episode_num):  
        if episode_num < 500:
            return np.random.randn(1,self.action_size)
        action +=  self.noise_decay_schedule(episode_num) * self.noise.sample()
        return action
                
        
    
    def reset(self):        
        self.noise.reset()
    
    def learn(self, agent_number, gamma,beta):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """                        
                                                  
        states, actions, rewards, next_states, dones = self.memory.sample()
        #states, actions, rewards, next_states, dones ,indices,weights = self.memory.sample(beta)       
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
                            
        #actions_next = self.actor_target(next_states)
        if agent_number == 0:   
            actions_next = self.actor_target(next_states[:,:self.state_size])
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:       
            actions_next = self.actor_target(next_states[:,self.state_size:])
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)
            
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)                        

        # Minimize the loss

        #critic_loss = (torch.FloatTensor(weights) * F.mse_loss(Q_expected, Q_targets)).mean()            
        critic_loss =  F.mse_loss(Q_expected, Q_targets)
        #huber_loss=torch.nn.SmoothL1Loss()        
        #critic_loss=huber_loss(Q_expected, Q_targets.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss                 
        
        if agent_number == 0:                    
            actions_pred = self.actor_local(states[:,:self.state_size])
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:            
            actions_pred = self.actor_local(states[:,self.state_size:])
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU) 
                           

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.13, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state        
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

    
class ReplayBuffer:    

    def __init__(self, batch_size, buffer_size, seed):
        """Initialize a ReplayBuffer object."""
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def is_filled(self):
        """Return the current size of internal memory."""
        return len(self.memory) >= self.batch_size