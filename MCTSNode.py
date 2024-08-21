import numpy as np

class MCTSNode:
    def __init__(self, state, environment, parent=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.environment = environment
    
    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def is_fully_expanded(self):
        print(f'Children in is fully expanded: {len(self.children)}')
        action_space, action_details, action_mask = self.environment.get_action_space()
        return len(self.children) == len(action_details)
    
    def expand(self, action, next_state, prior_prob):
        print(f"Expanding node with action: {action}")
        print(f"Prior probability: {prior_prob}")
        print(f"Next state: {next_state}")
        
        child_node = MCTSNode(state=next_state, environment=self.environment, parent=self, prior_prob=prior_prob)
        self.children[action] = child_node
        self.expanded = True  # Set the expanded flag
        
        print(f"Child node created. Children count: {len(self.children)}")
        print(f"Actions in children: {list(self.children.keys())}")
        
        return child_node
        
    def select(self, c_puct, valid_actions):
        best_action, best_child = None, None
        max_ucb = -float('inf')

        # Iterate over all children and filter based on valid actions
        for action, child in self.children.items():
            if action not in valid_actions:
                continue  # Skip invalid actions

            # Compute UCB value
            ucb = child.value + c_puct * child.prior_prob * np.sqrt(self.visit_count) / (1 + child.visit_count)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action, best_child = action, child

        return best_action, best_child
    
    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)


        