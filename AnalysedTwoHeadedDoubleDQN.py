import torch
import numpy as np
import matplotlib.pyplot as plt

class AnalysisTool:
    def __init__(self, dqn_model):
        self.dqn_model = dqn_model
        self.state_action_values = {}

    def explain_q_value(self, state, action):
        transformed_state = self.dqn_model.transform_state_for_nn(state)
        offensive_q, defensive_q, _ = self.dqn_model.online_network(transformed_state)
        q_value = offensive_q[0, action]
        return q_value.item()

    def compute_saliency_map(self, state, action):
        transformed_state = self.dqn_model.transform_state_for_nn(state)
        transformed_state.requires_grad_()
        offensive_q, _, _ = self.dqn_model.online_network(transformed_state)
        q_value = offensive_q[0, action]
        q_value.backward()
        return transformed_state.grad.abs().squeeze().numpy()

    def compare_actions(self, state, actions):
        explanations = []
        transformed_state = self.dqn_model.transform_state_for_nn(state)
        offensive_q, defensive_q, _ = self.dqn_model.online_network(transformed_state)
        for action in actions:
            explanations.append({
                'action': action,
                'offensive_q_value': offensive_q[0, action].item(),
                'defensive_q_value': defensive_q[0, action].item()
            })
        return explanations

    def plot_q_value_distribution(self, state, action_details):
        transformed_state = self.dqn_model.transform_state_for_nn(state)
        offensive_q, defensive_q, _ = self.dqn_model.online_network(transformed_state)
        q_values = offensive_q.detach().numpy()[0]

        plt.figure(figsize=(15, 5))
        plt.bar(range(len(q_values)), q_values)
        plt.xlabel('Action Index')
        plt.ylabel('Q-Value')
        plt.title('Q-Value Distribution Across Actions')
        plt.tight_layout()
        plt.savefig('q_value_distribution.png')
        plt.close()

        # Print top N actions
        top_n = 5
        top_indices = np.argsort(q_values)[-top_n:]
        for idx in reversed(top_indices):
            print(f"Action {idx}: Q-Value = {q_values[idx]}, Action = {action_details[idx]}")

    def record_state_action_value(self, state, action, offensive_reward, defensive_reward):
        key = (tuple(state.flatten()), action)
        if key not in self.state_action_values:
            self.state_action_values[key] = []
        self.state_action_values[key].append((offensive_reward, defensive_reward))

    def get_top_state_actions(self, n=10):
        avg_values = {k: (np.mean([r[0] for r in v]), np.mean([r[1] for r in v])) for k, v in self.state_action_values.items()}
        return sorted(avg_values.items(), key=lambda x: sum(x[1]), reverse=True)[:n]