import numpy as np
import torch

# transition model T[s1, a, s2] = Prob(s' | s, a)
T = np.random.dirichlet(np.ones(100), (100, 4)) 

class Environment:
    """
    A general class for defining a Markov Reward Process, containing:
        states: the set of possible states 
        rewards: the set of possible rewards
        actions: the set of possible actions 
    """
    def __init__(self,
                 states,
                 rewards,
                 actions):
        
        self.states = states
        self.rewards = rewards 
        self.actions = actions

class ObservableEnvironment(Environment):
    """
    An environment subclass with explicit dynamics, where 
        dynamics: a tensor of size SxRxSxA, with each entry a probability of transitioning 
                  to state s' and receiving reward r given a starting state s and action a 
    """
    def __init__(self,
                 states,
                 rewards,
                 actions,
                 dynamics,
                 reward_dynamics: bool = True):
        
        super().__init__(states,rewards,actions)
        self.dynamics = dynamics 
        self.reward_dynamics = reward_dynamics
    
    def __str__(self):
            return (
                f"ObservableEnvironment:\n"
                f"  States: {self.states.tolist()}\n"
                f"  Rewards: {self.rewards.tolist()}\n"
                f"  Actions: {self.actions}\n"
                f"  Dynamics Shape: {self.dynamics.shape}\n"
            )
    
    def expected_rewards(self):
        R = torch.einsum('ijkl,j -> kl', self.dynamics.float(), self.rewards.float()) 
        return R
    
    def reduced_dynamics(self):
        P = torch.sum(self.dynamics,dim=1)
        return P
    
    def _terminal(self):
        return self.states[-1]
    
    
#################################
# Generate a random environment #
#################################

class RandomMRP(ObservableEnvironment):
    def __init__(self,S,A):
        states = torch.arange(0,S)
        actions = torch.arange(0,A)

        # transition function
        P = np.random.dirichlet(np.ones(S), (S, A)) 
        dynamics = torch.tensor(P.transpose(0, 2, 1))
        rewards = []

        # reward function R[s, a] = Reward(s, a)
        R = np.random.rand(S, A)
        R = torch.tensor(R)

        self.reward_function = R
        super().__init__(states, rewards, actions, dynamics, reward_dynamics=False)