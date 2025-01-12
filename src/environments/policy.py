from .environment import Environment
import torch

class Policy:
    """
    A policy class that validates and represents a policy for an environment.

    Args:
        environment (Environment): 
            The environment on which the policy is defined. The environment must 
            have states and actions defined.
        policy (function): 
            A function that maps a state to a distribution over the action space. 
            For each state, the policy function should return a callable that takes 
            an action as input and outputs the probability of selecting that action.

    Raises:
        ValueError: If the probabilities for any state's action distribution do not sum to 1.
    """
    def __init__(self, environment: Environment, policy):
        self.environment = environment
        self.policy = policy

        # Validate the policy for each state
        for state in environment.states:
            action_probs = [policy(state)(action) for action in environment.actions]
            total_prob = sum(action_probs)
            
            # Check if the probabilities sum to 1 (with a small tolerance for floating-point errors)
            if abs(total_prob - 1) > 1e-6:
                raise ValueError(
                    f"Invalid policy: For state {state}, the probabilities over actions "
                    f"do not sum to 1. Total sum: {total_prob}."
                )

    def __call__(self, state):
        env = self.environment
        actions = env.actions
        probs = self.policy(state)

        return actions[torch.multinomial(probs, 1, replacement=True).item()]
    
    def _transition_environment(self, state):
        action = self(state)
        next_state,reward = torch.multinomial(self.environment, 1, replacement=True).item()
        return next_state,reward
