import numpy as np
np.bool8 = np.bool_

import gym

# Text (ANSI) render mode
env = gym.make("FrozenLake-v1", render_mode="ansi")
print("States:", env.observation_space.n)
print("Actions:", env.action_space.n)

state = env.reset()
action = env.action_space.sample()
new_state, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

# For text console output
out = env.render()
print(out)

# Full info
print("Transition:", state, "→", new_state,
      "action=", action,
      "reward=", reward,
      "done=", done,
      "info=", info)

env.close()

