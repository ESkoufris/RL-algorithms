from .environment import *
from .policy import *

import torch

class LinearWalk(ObservableEnvironment):
    """
    A simple environment representing a one-dimensional walk between discrete states.

    The agent starts in a state `s` and can perform one of two actions: 'left' or 'right'.
    Based on the chosen action and the current state, the agent transitions to a new state
    with deterministic dynamics. The environment also defines rewards associated with reaching
    certain terminal states.

    Attributes:
        states (torch.Tensor): A tensor representing the states in the environment, numbered from 1 to N.
        rewards (torch.Tensor): A tensor with rewards for the states. Rewards are -(N-s) for non-terminal states
                                and 0 for terminal states.
        actions (list): A list of available actions, ['left', 'right'].
        dynamics (torch.Tensor): A 4D tensor of shape (len(states), len(rewards), len(states), len(actions))
                                 representing the transition dynamics of the environment.
                                 
    Dynamics Details:
        - `dynamics[s_prime, r, s, a]` specifies the probability of transitioning from state `s` to `s_prime`
          with reward `r` after taking action `a`.

    Example:
        >>> env = SimpleWalk(N=5)
        >>> print(env.states)  # Output: tensor([1, 2, 3, 4, 5])
        >>> print(env.actions)  # Output: ['left', 'right']
        >>> print(env.dynamics.shape)  # Output: torch.Size([5, 2, 5, 2])

    Args:
        N (int): The number of states in the environment. States are numbered from 1 to N.
    """
    def __init__(self,N: int):
        if N <= 1:
            raise ValueError("The number of states (N) must be greater than 1.")
        
        states = torch.arange(0,N)
        rewards = states
        actions = ['left','stay','right']
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

class RandomWalkPolicy(Policy):
    def __init__(self, N):
        def policy(state):
            return 0.5*torch.ones(2)
        
        super().__init__(LinearWalk(N), policy)

#############
# Gridworld #
#############