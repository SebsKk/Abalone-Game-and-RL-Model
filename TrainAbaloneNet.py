
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

import torch
import numpy as np
import os
from datetime import datetime

def train_abalone_mcts(num_episodes, environment, model, n_mcts=300, c_puct=5, temp=1e-3, lr=0.001, model_save_interval=2, result_save_interval=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = []
    game_history = []
    
    # Create a directory for saving models and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"C:/Users/kaczm/Desktop/Abalone Project/Abalone in progress/models_abalone_net/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    for episode in range(num_episodes):
        environment.reset()
        print(f"\n--- Starting Episode {episode + 1} ---")
        print(f"Initial board state:\n{environment.game.board.grid}")
        print(f"Initial scores: Player 1 = {environment.game.players[0].score}, Player 2 = {environment.game.players[1].score}")
        
        mcts = MCTS(model.predict, environment, c_puct=c_puct, n_playout=n_mcts)
        states, mcts_probs, current_players = [], [], []
        episode_step = 0
        move_counter = 0
        
        while True:
            state = environment.get_current_state()
            action_space, action_details, action_mask = environment.get_action_space()
            print(f'\nMove {move_counter + 1}:')
            print(f'Current player: {environment.game.current_player.color}')
            print(f'Action space size: {len(action_space)}')
            
            if not action_space:
                print(f"No valid actions in episode {episode}. Ending episode.")
                break
            
            # MCTS action selection
            acts, probs = mcts.get_move_probs(state, temp=temp)
            
            if not acts:
                print(f"MCTS returned no valid actions in episode {episode}. Ending episode.")
                break
            
            action = select_action(acts, probs, action_details)
            
            if action is None:
                print(f"select_action returned None in episode {episode}. Ending episode.")
                break

            move_counter += 1
            print(f"Selected action: {action}")
            
            # Store the state and MCTS probabilities
            states.append(state)
            mcts_probs.append(probs)
            current_players.append(environment.game.current_player.color)
            
            # Make the move
            next_state, reward, done = environment.step(action)

            print(f'Board state after move:\n{environment.game.board.grid}')
            print(f'Scores after move: Player 1 = {environment.game.players[0].score}, Player 2 = {environment.game.players[1].score}')
            
            if environment.game.players[0].score != 0 or environment.game.players[1].score != 0:
                print("!!! Score changed !!!")
                print(f"Player 1 score: {environment.game.players[0].score}")
                print(f"Player 2 score: {environment.game.players[1].score}")
            
            done = environment.is_game_over()
            
            if done:
                winner = environment.get_winner().color
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                
                # Train on the game
                episode_loss = model.train(states, mcts_probs, winners_z)

                print(f"\n--- Episode {episode+1} Completed ---")
                print(f"Total Moves: {move_counter}")
                print(f"Final Scores: Player 1 = {environment.game.players[0].score}, Player 2 = {environment.game.players[1].score}")
                print(f"Winner: Player {winner}")
                print(f"Episode Loss: {episode_loss:.4f}")
                
                results.append((episode, episode_loss, winner))
                game_history.append((states, mcts_probs, winners_z))
                break
            
            elif move_counter >= 200:
                print(f"Episode {episode+1} reached max moves. Ending episode.")
                player_scores = [p.score for p in environment.game.players]
                
                if player_scores[0] > player_scores[1]:
                    winner = 1  # Player 1 wins
                elif player_scores[1] > player_scores[0]:
                    winner = -1  # Player 2 wins
                else:
                    winner = 0  # It's a tie

                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                
                # Train on the game
                episode_loss = model.train(states, mcts_probs, winners_z)

                print(f"\n--- Episode {episode+1} Completed (Max Moves) ---")
                print(f"Total Moves: {move_counter}")
                print(f"Final Scores: Player 1 = {environment.game.players[0].score}, Player 2 = {environment.game.players[1].score}")
                print(f"Winner: {'Player 1' if winner == 1 else 'Player 2' if winner == -1 else 'Tie'}")
                print(f"Episode Loss: {episode_loss:.4f}")
                
                results.append((episode, episode_loss, winner))
                game_history.append((states, mcts_probs, winners_z))
                break

            state = next_state
            episode_step += 1
            mcts.update_with_move(action)

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

    print(f'acts in select action: {acts} ')
    print(f'probs in select action: {probs} ')
    if not acts:
        print("No actions available. Returning None.")
        return None
    
    if isinstance(acts[0], int):  # If acts are indices
        valid_acts = [act for act in acts if act in action_details and action_details[act] is not None]
        actions = [action_details[act] for act in valid_acts]
        valid_probs = [p for act, p in zip(acts, probs) if act in action_details and action_details[act] is not None]
    else:
        actions = [act for act in acts if act is not None]
        valid_probs = [p for act, p in zip(acts, probs) if act is not None]

    if not actions:
        print("No valid actions after processing. Returning None.")
        return None

    if len(actions) != len(valid_probs):
        print(f"Mismatch between actions ({len(actions)}) and probabilities ({len(valid_probs)}). Adjusting...")
        min_len = min(len(actions), len(valid_probs))
        actions = actions[:min_len]
        valid_probs = valid_probs[:min_len]

    # Renormalize probabilities
    valid_probs = np.array(valid_probs)
    sum_probs = valid_probs.sum()
    if sum_probs == 0:
        print("All probabilities are zero. Choosing uniformly.")
        valid_probs = np.ones_like(valid_probs) / len(valid_probs)
    else:
        valid_probs /= sum_probs

    try:
        chosen_action = np.random.choice(actions, p=valid_probs)
        if chosen_action is None:
            print("Warning: Chosen action is None. Retrying selection.")
            return select_action(actions, valid_probs, action_details)  # Recursive call
        return chosen_action
    except ValueError as e:
        print(f"Error in np.random.choice: {e}")
        print(f"Actions: {actions}")
        print(f"Probabilities: {valid_probs}")
        return None

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

    

    game_ops_rl = GameOpsRL(player1, player2, 400)
    
    input_dim = 243  # 9x9x3 for the board representation
    output_dim = 140 
    
    model = AbaloneNet(input_dim, output_dim, game_ops_rl)
    
    results, game_history = train_abalone_mcts(100, game_ops_rl, model)