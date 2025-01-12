from .environment import *
from .policy import *

import torch

class Gridworld(ObservableEnvironment):
    """
    """
    def __init__(self,N: int):
        if N <= 1:
            raise ValueError("The number of states (N) must be greater than 1.")
        
        states = torch.arange(0,N*N)
        rewards = torch.tensor([0,-1,10])
        actions = ['left','stay','right','up','down']
        dynamics = torch.zeros((len(states),len(rewards),len(states),len(actions)))

        for s in states:
            if 0 < s < N-1:
                # moving left
                dynamics[s-1, s-1, s, 0] = 1
                #staying put 
                dynamics[s, s, s, 1] = 1 
                # moving right
                dynamics[s+1, s+1, s, 2] = 1
            elif s == 0:
                # # moving left 
                dynamics[s,s,s,0] = 1
                # staying put
                dynamics[s,s,s,1] = 1
                # moving right
                dynamics[s+1,s+1,s,2] = 1
            elif s == N-1:
                # moving left
                dynamics[s-1,s-1,s,0] = 1 
                # staying put 
                dynamics[s,s,s,1] = 1 
                # moving right 
                dynamics[s,s,s,2] = 1 
        
        super().__init__(states,rewards,actions,dynamics)