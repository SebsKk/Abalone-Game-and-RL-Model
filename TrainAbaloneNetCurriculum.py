
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from AbaloneNet import AbaloneNet, MCTS
from GameOpsRL import GameOpsRL
from Player import Player
from CurriculumLearning import CurriculumLearning

def train_abalone_mcts(num_episodes, environment, model, curriculum, n_mcts=1000, c_puct=5, temp=1e-3, lr=0.001, model_save_interval=100, result_save_interval=40):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = []
    game_history = []
    
    # Create a directory for saving models and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"C:/Users/kaczm/Desktop/Abalone Project/Abalone in progress/models_abalone_net/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    for episode in range(num_episodes):
        environment.reset()
        difficulty = curriculum.get_current_difficulty()
        environment.game.board = curriculum.adjust_board(environment.game.board)
        environment.max_moves = difficulty['max_moves']
        
        mcts = MCTS(model.predict, environment, c_puct=c_puct, n_playout=n_mcts)
        states, mcts_probs, current_players = [], [], []
        episode_step = 0
        
        while True:
            state = environment.get_current_state()
            action_space, action_details, action_mask = environment.get_action_space()
            
            # MCTS action selection
            acts, probs = mcts.get_move_probs(state, temp=temp)
            action = select_action(acts, probs, action_details)
            
            # Store the state and MCTS probabilities
            states.append(state)
            mcts_probs.append(probs)
            current_players.append(environment.game.current_player.color)
            
            # Make the move
            next_state, _, done = environment.step(action)
            
            # Update curriculum and check for game end
            curriculum.increment_move_count()
            done = curriculum.is_game_over(environment)
            
            if done:
                # Get game result
                winner = environment.game.winner
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                
                # Train on the game
                episode_loss = model.train(states, mcts_probs, winners_z)
                results.append((episode, episode_loss, winner))
                game_history.append((states, mcts_probs, winners_z))
                
                print(f"Episode {episode+1}: Loss = {episode_loss:.4f}, Winner = {winner}, Moves = {episode_step}, Difficulty = {curriculum.current_level}")
                break
            
            state = next_state
            episode_step += 1
            mcts.update_with_move(action)
        
        # Update curriculum
        curriculum.update(winner, episode_step, True)
        
        # Save results and model periodically
        if (episode + 1) % result_save_interval == 0:
            save_results(save_dir, episode + 1, results)
        
        if (episode + 1) % model_save_interval == 0:
            model_path = os.path.join(save_dir, f"abalone_mcts_model_episode_{episode+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at episode {episode+1}")

    # Save final model and results
    final_model_path = os.path.join(save_dir, "final_abalone_mcts_model.pth")
    torch.save(model.state_dict(), final_model_path)
    save_results(save_dir, num_episodes, results)
    plot_results(results, save_dir)

    print(f"Training completed. Final results and model saved in {save_dir}")
    return results, game_history

def select_action(acts, probs, action_details):
    if isinstance(acts[0], int):  # If acts are indices
        #print(f'Action space: {acts}, action details: {action_details}')
        actions = [action_details[act] for act in acts]
    else:  # If acts are already action objects
        actions = acts
    return np.random.choice(actions, p=probs)

def save_results(save_dir, episode, results):
    results_path = os.path.join(save_dir, f"results_episode_{episode}.json")
    with open(results_path, 'w') as f:
        json.dump({
            "episodes": [r[0] for r in results],
            "losses": [r[1] for r in results],
            "winners": [r[2] for r in results]
        }, f)
    print(f"Results saved up to episode {episode}")

def plot_results(results, save_dir):
    episodes, losses, winners = zip(*results)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episodes, losses)
    plt.title("Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    winner_counts = {1: 0, -1: 0, 0: 0}
    for winner in winners:
        winner_counts[winner] += 1
    plt.bar(winner_counts.keys(), winner_counts.values())
    plt.title("Game Outcomes")
    plt.xlabel("Winner")
    plt.ylabel("Count")
    plt.xticks([-1, 0, 1], ['Player 2', 'Draw', 'Player 1'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_results.png"))
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
    
    input_dim = 243  # 9x9x3 for the board representation
    output_dim = 140 
    
    model = AbaloneNet(input_dim, output_dim, game_ops_rl)
    
    results, game_history = train_abalone_mcts(10, game_ops_rl, model, curriculum)