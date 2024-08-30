from TwoHeadedDoubleDQN import TwoHeadedDoubleDQN

from Player import Player
from GameOpsRL import GameOpsRL
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from CurriculumLearning import CurriculumLearning
from RewardSystemTwoHeadedSimplified import RewardSystemTwoHeadedSimplified
from RewardSystemTwoHeadedSimplifiedALot import RewardSystemTwoHeadedSimplifiedALot
from AnalysisTool import AnalysisTool


def train_dqn(num_episodes, environment, dqn_model, reward_system, curriculum, epsilon_start=1.0, epsilon_decay=0.99998, epsilon_min=0.03, reward_save_interval=40, model_save_interval=100, last_saved_episode=0):
    epsilon = epsilon_start
    results = []
    action_history = {}
    episode_reward_counters = []
    
    # Create a directory for saving models and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"C:/Users/kaczm/Desktop/Abalone Project/Abalone in progress/models_deepq/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    #analysis_tool = AnalysisTool(dqn_model)
    #analysis_tool.set_save_directory(save_dir)

    for episode in range(num_episodes):
        environment.reset()
        difficulty = curriculum.get_current_difficulty()
        environment.game.board = curriculum.adjust_board(environment.game.board)
        environment.max_moves = difficulty['max_moves']
        state = environment.get_current_state()

        total_offensive_reward = 0
        total_defensive_reward = 0
        done = False
        episode_actions = []
        curriculum.reset_move_count()
        reward_system.move_count = 0
        episode_counters = reward_system.get_and_reset_episode_counters()
        episode_reward_counters.append(episode_counters)

        while not done:
            action_space, action_details, action_mask = environment.get_action_space()
            transformed_state = dqn_model.transform_state_for_nn(state)
            action, action_index = dqn_model.choose_action(transformed_state, epsilon, action_space, action_details)

            print(f"Action chosen: {action}")
            
            episode_actions.append(action)
            next_state, move_valid, done, ball_pushed_off = environment.step(action)
            
            curriculum.increment_move_count()
            done = curriculum.is_game_over(environment)
            
            offensive_reward = reward_system.calculate_offensive_reward(state, next_state, action['start'], action['end'])
            defensive_reward = reward_system.calculate_defensive_reward(state, next_state, action['start'], action['end'])
            
            print(f' offensive reward from move is {offensive_reward} and defensive reward is {defensive_reward}')
            total_offensive_reward += offensive_reward
            total_defensive_reward += defensive_reward
            
            transformed_next_state = dqn_model.transform_state_for_nn(next_state)
            # encoded_action = environment.encode_action(action)
            
            next_action_space, next_action_details, next_action_mask = environment.get_action_space()
            
            #analysis_tool.record_state_action_value(state, encoded_action, offensive_reward, defensive_reward)
            dqn_model.update(transformed_state, action_index, offensive_reward, defensive_reward, transformed_next_state, action_space, next_action_space, done)
            
            print(f'DQN update done for episode {episode+1}')
            if episode % 100 == 0:  
                dqn_model.update_target_network()
                '''# Perform analysis at the end of the episode
                action_space, action_details, action_mask = environment.get_action_space()
                
                # Explain Q-value for the chosen action
                explanation = analysis_tool.explain_q_value(state, encoded_action, episode)
                
                # Compute and save saliency map
                saliency_map = analysis_tool.compute_saliency_map(state, encoded_action, episode)
                
                # Compare top actions
                top_actions = action_space[:5]  # Assume action_space is sorted
                comparisons = analysis_tool.compare_actions(state, top_actions, episode)
                
                # Plot Q-value distribution
                analysis_tool.plot_q_value_distribution(state, action_details, episode)'''

            state = next_state
            print(f'Next state is {state}')
        
        print('Episode done')
        
        curriculum.update((total_offensive_reward + total_defensive_reward) / 2, curriculum.move_count, True)
  
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        results.append((total_offensive_reward, total_defensive_reward))
        action_history[episode] = episode_actions
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}: Offensive Reward = {total_offensive_reward}, Defensive Reward = {total_defensive_reward}, Epsilon = {epsilon:.4f}, Difficulty Level = {curriculum.current_level}")
        
        if (episode + 1) % reward_save_interval == 0:
            save_rewards_and_actions(save_dir, episode + 1, results, epsilon, action_history, last_saved_episode)
            last_saved_episode = episode
        
        if (episode + 1) % model_save_interval == 0:
            model_path = os.path.join(save_dir, f"two_headed_dqn_model_episode_{episode+1}.pth")
            torch.save(dqn_model.state_dict(), model_path)
            print(f"Model saved at episode {episode+1}")

    plot_results(results, save_dir)

    final_model_path = os.path.join(save_dir, "final_two_headed_dqn_model.pth")
    torch.save(dqn_model.state_dict(), final_model_path)
    save_rewards_and_actions(save_dir, num_episodes, results, epsilon, action_history, last_saved_episode)

    #top_state_actions = analysis_tool.get_top_state_actions()
    #analysis_tool.save_final_analysis()
    print(f"Training completed. Final results and model saved in {save_dir}")
    return results, action_history, episode_reward_counters

def save_rewards_and_actions(save_dir, episode, results, epsilon, action_history, last_saved_episode):
    new_episodes = list(range(last_saved_episode + 1, episode + 1))
    new_results = results[last_saved_episode:]
    
    results_path = os.path.join(save_dir, f"results_episode_{episode}.json")
    with open(results_path, 'w') as f:
        json.dump({
            "episodes": new_episodes,
            "offensive_rewards": [r[0] for r in new_results],
            "defensive_rewards": [r[1] for r in new_results],
            "current_epsilon": epsilon
        }, f)
    
    new_action_history = {str(ep): actions for ep, actions in action_history.items() if ep > last_saved_episode}
    
    action_history_path = os.path.join(save_dir, f"action_history_episode_{episode}.json")
    with open(action_history_path, 'w') as f:
        json.dump(new_action_history, f)

    print(f"Rewards and action history from episode {last_saved_episode + 1} to {episode} saved in {save_dir}")

def plot_results(results, save_dir):
    plt.figure(figsize=(10, 5))
    offensive_rewards = [r[0] for r in results]
    defensive_rewards = [r[1] for r in results]
    plt.plot(offensive_rewards, label='Offensive Rewards')
    plt.plot(defensive_rewards, label='Defensive Rewards')
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "reward_plot.png"))
    plt.close()

def plot_reward_usage(episode_reward_counters, save_dir):
    # Separate offensive and defensive counters
    offensive_counters = {key: [] for key in episode_reward_counters[0]['offensive']}
    defensive_counters = {key: [] for key in episode_reward_counters[0]['defensive']}

    for episode_counter in episode_reward_counters:
        for key in offensive_counters:
            offensive_counters[key].append(episode_counter['offensive'][key])
        for key in defensive_counters:
            defensive_counters[key].append(episode_counter['defensive'][key])

    # Plot offensive rewards
    plt.figure(figsize=(15, 10))
    for key, values in offensive_counters.items():
        plt.plot(values, label=key)
    plt.title("Offensive Reward Usage Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "offensive_reward_usage.png"))
    plt.close()

    # Plot defensive rewards
    plt.figure(figsize=(15, 10))
    for key, values in defensive_counters.items():
        plt.plot(values, label=key)
    plt.title("Defensive Reward Usage Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "defensive_reward_usage.png"))
    plt.close()

    # Plot total rewards (offensive + defensive) per episode
    total_rewards = []
    for episode_counter in episode_reward_counters:
        total_offensive = sum(episode_counter['offensive'].values())
        total_defensive = sum(episode_counter['defensive'].values())
        total_rewards.append(total_offensive + total_defensive)

    plt.figure(figsize=(15, 5))
    plt.plot(total_rewards)
    plt.title("Total Reward Usage Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Count")
    plt.savefig(os.path.join(save_dir, "total_reward_usage.png"))
    plt.close()

    # Plot reward type distribution
    offensive_totals = {key: sum(values) for key, values in offensive_counters.items()}
    defensive_totals = {key: sum(values) for key, values in defensive_counters.items()}

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.pie(offensive_totals.values(), labels=offensive_totals.keys(), autopct='%1.1f%%')
    plt.title("Offensive Reward Distribution")
    plt.subplot(1, 2, 2)
    plt.pie(defensive_totals.values(), labels=defensive_totals.keys(), autopct='%1.1f%%')
    plt.title("Defensive Reward Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_distribution.png"))
    plt.close()

    # Calculate and print statistics
    print("Offensive Reward Statistics:")
    for key, values in offensive_counters.items():
        print(f"{key}: Mean = {np.mean(values):.2f}, Std = {np.std(values):.2f}, Max = {np.max(values)}, Min = {np.min(values)}")

    print("\nDefensive Reward Statistics:")
    for key, values in defensive_counters.items():
        print(f"{key}: Mean = {np.mean(values):.2f}, Std = {np.std(values):.2f}, Max = {np.max(values)}, Min = {np.min(values)}")

if __name__ == "__main__":
    player1 = Player("Black", 1)
    player2 = Player("White", -1)
    
    difficulty_levels = [
        {"max_moves": 40, "pieces_per_player": 6},
        {"max_moves": 60, "pieces_per_player": 9},
        {"max_moves": 100, "pieces_per_player": 12},
        {"max_moves": 250, "pieces_per_player": 14}
    ]
    
    curriculum = CurriculumLearning(difficulty_levels)
    initial_difficulty = curriculum.get_current_difficulty()
    
    game_ops_rl = GameOpsRL(player1, player2, initial_difficulty['max_moves'])
    game_ops_rl.game.board = curriculum.adjust_board(game_ops_rl.game.board)
    
    dqn_model = TwoHeadedDoubleDQN(243, 1686, game_ops_rl)
    # reward_system = RewardSystemTwoHeaded(player1, player2)
    
    reward_system = RewardSystemTwoHeadedSimplifiedALot(player1, player2)

    results, action_history, episode_reward_counters = train_dqn(100000, game_ops_rl, dqn_model, reward_system, curriculum)