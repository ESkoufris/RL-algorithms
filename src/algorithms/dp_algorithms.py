import matplotlib.pyplot as plt
from src.environments.environment import *
from src.environments.policy import *

###################
# VALUE ITERATION #
###################
def value_iteration(env, tol=0.1, gamma=0.9):
    """
    Perform Value Iteration to compute the optimal value function and policy,
    while plotting max_diff live during each iteration.

    Args:
        env (Environment): The environment with states, dynamics, and rewards.
        tol (float): Convergence threshold for value function updates.
        gamma (float): Discount factor for future rewards.

    Returns:
        value_functions (torch.Tensor): All intermediate value functions (stacked).
        pi_star (torch.Tensor): The optimal policy for each state.
    """
    states = env.states
    P = env.dynamics.float()
    # rewards = env.rewards.float()

    value_functions = []
    V = torch.zeros(len(states)).float()
    value_functions.append(V)

    # compute reward function R(s,a)
    if env.reward_dynamics:
        # remove R dynamics 
        R = env.expected_rewards().float()
        P = env.reduced_dynamics().float()
    else:
        R = env.reward_function.float()
        P = env.dynamics.float()
    
    max_diffs = []  

    # Initialize the live plot
    plt.ion()  
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], marker='o', linestyle='-', color='b', label="Max Difference")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max Difference (max_diff)")
    ax.set_title("Live Value Iteration Convergence")
    ax.grid(True)
    ax.legend()

    iteration = 0

    while True:
        q_update = R + gamma*torch.einsum('sxa,s -> xa', P, V)
        W = torch.max(q_update, dim=1).values
        max_diff = torch.max(torch.abs(W - V)).item()
        max_diffs.append(max_diff)  # Append max_diff to the list
        V = W
        value_functions.append(V)

        # Update the live plot during the loop
        line.set_xdata(range(len(max_diffs)))
        line.set_ydata(max_diffs)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()  # Force a redraw of the canvas
        fig.canvas.flush_events()  # Ensure the events are processed

        iteration += 1
        if max_diff < tol:
            break

    plt.ioff()  # Turn off interactive mode
    plt.show()

    Q = R + gamma*torch.einsum('sxa,s -> xa', P, V)
    pi_star = torch.argmax(Q, dim=1)
    return [V, pi_star]

####################
# POLICY ITERATION #
####################
def policy_iteration(tol, policy: Policy):
    pass

