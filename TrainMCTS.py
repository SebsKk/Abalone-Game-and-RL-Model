import os
import torch
import datetime
from collections import deque
from GameOpsRL import GameOpsRL
from CNNPolicyValueNetwork import CNNPolicyValueNetwork
from MCTS import MCTS
from Player import Player



# Assuming you have GameOpsRL, CNNPolicyValueNetwork, and MCTS already defined
# Initialize environment, neural network, and MCTS
player1 = Player("Black", 1)
player2 = Player("White", -1)

    

environment = GameOpsRL(player1, player2)  # Your game environment

num_actions = 140

# Create the CNN and MCTS instances
cnn = CNNPolicyValueNetwork(board_size=9, num_actions=num_actions)
mcts = MCTS(neural_network=cnn, environment=environment)

# Set the training parameters
num_episodes = 1  # Total number of episodes to run
save_interval = 1  # Save every 10 episodes
learning_rate = 0.001  # Learning rate for optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Tracking results
episode_rewards = []
wins = 0
losses = 0
draws = 0

# Create the model directory if it doesn't exist
model_save_path = "C:/Users/kaczm/Desktop/Abalone Project/Abalone in progress/models_abalone_net"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Training loop
for episode in range(1, num_episodes + 1):
    state = environment.reset()
    episode_reward = 0
    done = False
    trajectory = []  # To store the states and actions for the episode
    
    # Play one episode
    while not done:
        # Transform state for neural network input
        transformed_state = cnn.transform_state_for_nn(state)
        
        # Use MCTS to select an action
        curr_state = environment.get_current_state()
        action = mcts.run(transformed_state, curr_state)
        
        # Store the state and action in the trajectory
        trajectory.append((transformed_state, action))
        
        # Take a step in the environment
        next_state, _, done = environment.step(action)
        
        # Update state
        state = next_state

    # Determine the winner and assign a final reward
    winner = environment.get_winner()  # Method to determine the winner
    if winner == environment.current_player:  # Assuming you have a way to check the winner
        final_reward = 1  # Win
        wins += 1
    elif winner == -environment.current_player:  # Opponent wins
        final_reward = -1  # Loss
        losses += 1
    else:
        final_reward = 0  # Draw
        draws += 1

    # Backpropagate through the entire trajectory
    optimizer.zero_grad()
    for state, action in trajectory:
        # Pass the state to the network
        transformed_state_tensor = torch.tensor(state).unsqueeze(0).float()
        policy_probs, value = cnn.predict(transformed_state_tensor)
        
        # Compute loss based on the final reward
        # Value loss: Compare predicted value with final_reward
        value_loss = F.mse_loss(value, torch.tensor([final_reward]).float())
        
        # Policy loss: Encourage the network to predict actions that lead to the outcome
        action_prob = policy_probs[action]
        policy_loss = -torch.log(action_prob) * final_reward  # Maximize the probability of winning actions
        
        # Combine losses
        loss = value_loss + policy_loss
        
        # Backpropagate the loss
        loss.backward()

    # Optimize the network
    optimizer.step()

    # Track episode rewards
    episode_rewards.append(episode_reward)

    # Print the status every episode
    print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward}, Wins: {wins}, Losses: {losses}, Draws: {draws}")

    # Save the model and results every save_interval episodes
    if episode % save_interval == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(model_save_path, f"abalone_net_{timestamp}.pth")
        
        # Save the model state dictionary
        torch.save(cnn.state_dict(), save_path)
        
        # Optionally, save the episode rewards to a file
        result_save_path = os.path.join(model_save_path, f"abalone_results_{timestamp}.txt")
        with open(result_save_path, "w") as f:
            f.write(f"Episode: {episode}\n")
            f.write(f"Total Wins: {wins}\n")
            f.write(f"Total Losses: {losses}\n")
            f.write(f"Total Draws: {draws}\n")
            f.write(f"Episode Rewards: {episode_rewards}\n")
        
        print(f"Model and results saved to {save_path}")
