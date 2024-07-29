from DQN import DQN
from RewardSystem import RewardSystem
from Player import Player
from GameOpsRL import GameOpsRL
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from CurriculumLearning import CurriculumLearning

def train_dqn(num_episodes, environment, dqn_model, reward_system, curriculum, epsilon_start=1.0, epsilon_decay=0.9995, epsilon_min=0.03, reward_save_interval=40, model_save_interval=100, last_saved_episode=0):
    epsilon = epsilon_start
    results = []
    action_history = {}
    
    # Create a directory for saving models and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"C:/Users/kaczm/Desktop/Abalone Project/Abalone in progress/models_deepq/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    for episode in range(num_episodes):

        state = environment.reset()
        difficulty = curriculum.get_current_difficulty()
        environment.game.board = curriculum.adjust_board(environment.game.board)
        environment.max_moves = difficulty['max_moves']

        total_reward = 0
        done = False
        episode_actions = []
        curriculum.reset_move_count()

        while not done:
            action_space, action_details, action_mask = environment.get_action_space()
            transformed_state = dqn_model.transform_state_for_nn(state)
            action, action_index = dqn_model.choose_action(transformed_state, epsilon, action_space, action_mask, action_details)

            print(f"Action chosen: {action}")
            
            episode_actions.append(action)
            next_state, _ = environment.step(action)  # Ignore the 'done' returned by step
            
            curriculum.increment_move_count()  # Increment move count after each action
            
            # Use curriculum to determine if the game is over
            done = curriculum.is_game_over(environment)
            
            reward = reward_system.calculate_reward(state, next_state, action['start'], action['end'])
            total_reward += reward
            
            transformed_next_state = dqn_model.transform_state_for_nn(next_state)
            encoded_action = environment.encode_action(action)
            
            next_action_space, next_action_details, next_action_mask = environment.get_action_space()
            
            dqn_model.update(transformed_state, encoded_action, reward, transformed_next_state, action_mask, next_action_mask, done)
            
            print(f'DQN update done for episode {episode+1}')

            state = next_state
        
        print('Episode done')
        
        # Update curriculum
        curriculum.update(total_reward, curriculum.move_count, True)
  
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        results.append(total_reward)
        action_history[episode] = episode_actions
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}: Total Reward = {total_reward}, Epsilon = {epsilon:.4f}, Difficulty Level = {curriculum.current_level}")
        
        # Save rewards and action history every 40 episodes
        if (episode + 1) % reward_save_interval == 0:
            save_rewards_and_actions(save_dir, episode + 1, results, epsilon, action_history, last_saved_episode)
            last_saved_episode = episode
        
        # Save model every 100 episodes
        if (episode + 1) % model_save_interval == 0:
            model_path = os.path.join(save_dir, f"dqn_model_episode_{episode+1}.pth")
            torch.save(dqn_model.state_dict(), model_path)
            print(f"Model saved at episode {episode+1}")

    # Plot results
    plot_results(results, save_dir)

    # Save final model, rewards, and action history
    final_model_path = os.path.join(save_dir, "final_dqn_model.pth")
    torch.save(dqn_model.state_dict(), final_model_path)
    save_rewards_and_actions(save_dir, num_episodes, results, epsilon, action_history, last_saved_episode)

    print(f"Training completed. Final results and model saved in {save_dir}")
    return results, action_history

def save_rewards_and_actions(save_dir, episode, results, epsilon, action_history, last_saved_episode):
    new_episodes = list(range(last_saved_episode + 1, episode + 1))
    new_results = results[last_saved_episode:]
    
    results_path = os.path.join(save_dir, f"results_episode_{episode}.json")
    with open(results_path, 'w') as f:
        json.dump({
            "episodes": new_episodes,
            "rewards": new_results,
            "current_epsilon": epsilon
        }, f)
    
    new_action_history = {str(ep): actions for ep, actions in action_history.items() if ep > last_saved_episode}
    
    action_history_path = os.path.join(save_dir, f"action_history_episode_{episode}.json")
    with open(action_history_path, 'w') as f:
        json.dump(new_action_history, f)

    print(f"Rewards and action history from episode {last_saved_episode + 1} to {episode} saved in {save_dir}")

def plot_results(results, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(results)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(save_dir, "reward_plot.png"))
    plt.close()


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
    
    dqn_model = DQN(243, 140, game_ops_rl)
    reward_system = RewardSystem(player1, player2)

    results, action_history = train_dqn(4000, game_ops_rl, dqn_model, reward_system, curriculum)