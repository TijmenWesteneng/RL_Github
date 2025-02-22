import numpy as np
import matplotlib.pyplot as plt


def visualise_policy(policy, title=""):
    policy_matrix = np.flip(policy.reshape(8,8), axis=0)

    arrows = {0:(-1,0), 1:(0,-1), 2:(1,0), 3:(0,1)} # For each action, we define the x and y direction of the arrow (e.g., for action 0, i.e. left, the arrow should point in direction (-1, 0))
    scale = 0.3

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xticks(np.arange(8) - 0.5)
    ax.set_yticks(np.arange(8) - 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)

    for r in range(8):
        for c in range(8):
            action = policy_matrix[r, c]
            dx, dy = arrows[action] # Get the arrow x and y directions
            ax.arrow(c, r, dx * scale, dy * scale, head_width=0.2, head_length=0.2, fc='blue', ec='blue') # create an arrow for the cell 

    ax.set_title(title)
    ax.set_aspect('equal') # ensure that scaling of x and y is equal so the grid remains square
    plt.show()


def visualise_values(values, title=""):
    value_matrix = np.flip(values.reshape(8,8), axis=0)

    ig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xticks(np.arange(8) - 0.5)
    ax.set_yticks(np.arange(8) - 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)

    for r in range(8):
        for c in range(8):
            value = value_matrix[r, c].round(2)
            ax.text(c, r, value, ha='center', va='center')

    ax.set_title(title)
    ax.set_aspect('equal') # ensure that scaling of x and y is equal so the grid remains square
    plt.show()


    