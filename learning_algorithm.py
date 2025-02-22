import matplotlib.pyplot as plt
import numpy as np

class LearningAlgorithm:
    def __init__(self):
        #INITIALIZE CLASS VAR
        self.trained = False
        self.value_function = None
        self.policy = None

    def run_training(self):
        pass

    def get_training_results(self):
        #If model has not been trained
        if not self.trained:
            self.run_training()
            self.trained = True
        
        return self.value_function, self.policy
    
    def visualise_policy(self):
        policy_matrix = self.policy.reshape(8,8)

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
                ax.arrow(c, 7 - r, dx * scale, dy * scale, head_width=0.2, head_length=0.2, fc='blue', ec='blue') # create an arrow for the cell 

        ax.set_title("Final policy")
        ax.set_aspect('equal') # ensure that scaling of x and y is equal so the grid remains square
        plt.show()