import gym
import numpy as np
import time
import matplotlib.pyplot as plt

# Fix for numpy compatibility
np.bool8 = np.bool_

# Create environment with correct name and render mode
env = gym.make('FrozenLake-v1', render_mode='ansi')  # Changed from v0 to v1
STATES = env.observation_space.n
ACTIONS = env.action_space.n

# Initialize Q-table
Q = np.zeros((STATES, ACTIONS))

# Hyperparameters
EPISODES = 1500
MAX_STEPS = 100
LEARNING_RATE = 0.81
GAMMA = 0.96
RENDER = False

# Epsilon-greedy parameters
epsilon = 0.9
rewards = []

for episode in range(EPISODES):
    # Reset returns a tuple in newer versions, we need just the state
    state, _ = env.reset()  # Fixed: unpack the tuple
    
    for _ in range(MAX_STEPS):
        if RENDER:
            rendered = env.render()
            if rendered:
                print(rendered)
        
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax(Q[state, :])
        
        # Take action - newer gym versions return 5 values
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Combine the two boolean flags
        
        # Q-learning update
        Q[state, action] = Q[state, action] + LEARNING_RATE * (
            reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]
        )
        
        state = next_state
        
        if done: 
            rewards.append(reward)
            epsilon = max(0.01, epsilon - 0.001)  # Prevent epsilon from going negative
            break

env.close()

# Results
print("Q-table:")
print(Q)
print(f"Average reward: {sum(rewards)/len(rewards):.3f}")

# Plot training progress
def get_average(values):
    return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
    if i+100 <= len(rewards):
        avg_rewards.append(get_average(rewards[i:i+100])) 

if avg_rewards:
    plt.plot(avg_rewards)
    plt.ylabel('Average Reward')
    plt.xlabel('Episodes (100\'s)')
    plt.title('Q-Learning Training Progress')
    plt.show()
else:
    print("Not enough episodes to plot averages")
