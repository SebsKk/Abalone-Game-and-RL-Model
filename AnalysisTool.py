import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class AnalysisTool:
    def __init__(self, dqn_model):
        self.dqn_model = dqn_model
        self.state_action_values = {}
        self.save_dir = None

    def set_save_directory(self, main_save_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(main_save_dir, f"analysis_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

    def explain_q_value(self, state, action, episode):
        transformed_state = self.dqn_model.transform_state_for_nn(state)
        offensive_q, defensive_q, _ = self.dqn_model.online_network(transformed_state)
        q_value = offensive_q[0, action]
        
        explanation = {
            "episode": episode,
            "action": action,
            "q_value": q_value.item()
        }
        
        with open(os.path.join(self.save_dir, f"q_value_explanation_episode_{episode}.json"), 'w') as f:
            json.dump(explanation, f)
        
        return q_value.item()

    def compute_saliency_map(self, state, action, episode):
        transformed_state = self.dqn_model.transform_state_for_nn(state)
        transformed_state.requires_grad_()
        offensive_q, _, _ = self.dqn_model.online_network(transformed_state)
        q_value = offensive_q[0, action]
        q_value.backward()
        saliency_map = transformed_state.grad.abs().squeeze().numpy()
        
        np.save(os.path.join(self.save_dir, f"saliency_map_episode_{episode}.npy"), saliency_map)
        
        return saliency_map

    def compare_actions(self, state, actions, episode):
        explanations = []
        transformed_state = self.dqn_model.transform_state_for_nn(state)
        offensive_q, defensive_q, _ = self.dqn_model.online_network(transformed_state)
        for action in actions:
            explanations.append({
                'action': action,
                'offensive_q_value': offensive_q[0, action].item(),
                'defensive_q_value': defensive_q[0, action].item()
            })
        
        with open(os.path.join(self.save_dir, f"action_comparison_episode_{episode}.json"), 'w') as f:
            json.dump(explanations, f)
        
        return explanations

    def plot_q_value_distribution(self, state, action_details, episode):
        transformed_state = self.dqn_model.transform_state_for_nn(state)
        offensive_q, defensive_q, _ = self.dqn_model.online_network(transformed_state)
        q_values = offensive_q.detach().numpy()[0]

        plt.figure(figsize=(15, 5))
        plt.bar(range(len(q_values)), q_values)
        plt.xlabel('Action Index')
        plt.ylabel('Q-Value')
        plt.title(f'Q-Value Distribution Across Actions (Episode {episode})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"q_value_distribution_episode_{episode}.png"))
        plt.close()

        top_n = 5
        top_indices = np.argsort(q_values)[-top_n:]
        top_actions = []
        for idx in reversed(top_indices):
            top_actions.append({
                "action_index": idx,
                "q_value": q_values[idx],
                "action_details": action_details[idx]
            })
        
        with open(os.path.join(self.save_dir, f"top_actions_episode_{episode}.json"), 'w') as f:
            json.dump(top_actions, f)

    def record_state_action_value(self, state, action, offensive_reward, defensive_reward):
        key = (tuple(state.flatten()), action)
        if key not in self.state_action_values:
            self.state_action_values[key] = []
        self.state_action_values[key].append((offensive_reward, defensive_reward))

    def get_top_state_actions(self, n=10):
        avg_values = {k: (np.mean([r[0] for r in v]), np.mean([r[1] for r in v])) for k, v in self.state_action_values.items()}
        top_state_actions = sorted(avg_values.items(), key=lambda x: sum(x[1]), reverse=True)[:n]
        
        with open(os.path.join(self.save_dir, "top_state_actions.json"), 'w') as f:
            json.dump([{
                "state": state.tolist(),
                "action": action,
                "avg_offensive_reward": off_reward,
                "avg_defensive_reward": def_reward
            } for ((state, action), (off_reward, def_reward)) in top_state_actions], f)
        
        return top_state_actions

    def save_final_analysis(self):
        with open(os.path.join(self.save_dir, "final_state_action_values.json"), 'w') as f:
            json.dump({
                str(k): v for k, v in self.state_action_values.items()
            }, f)