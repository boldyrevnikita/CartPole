# CartPole

This project implements a reinforcement learning (RL) agent using the REINFORCE algorithm to solve the CartPole-v1 environment. The implementation leverages PyTorch for building and training the neural network and OpenAI Gym for the environment.

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Code Structure](#code-structure)
5. [Hyperparameters](#hyperparameters)
6. [Results](#results)
7. [Visualization](#visualization)

---

## Overview

The goal of this project is to train an RL agent to balance a pole on a moving cart using the REINFORCE algorithm (a policy gradient method). The agent learns by interacting with the environment, receiving rewards, and updating its policy to maximize cumulative rewards.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook
- Libraries:
  - `gym`
  - `numpy`
  - `torch`
  - `matplotlib`
  - `pyvirtualdisplay`
  - `pygame`

### Steps
1. Install required system dependencies:
```
sudo apt install python-opengl ffmpeg xvfb
```
2. Install required Python packages:
```
pip install gym pyvirtualdisplay pyglet==1.5.1 matplotlib torch imageio
```

---

## Usage

1. Clone this repository and navigate to the project directory.
2. Open the provided Jupyter Notebook (`CartPole.ipynb`).
3. Run all cells sequentially to:
- Set up the environment.
- Define and train the policy network.
- Evaluate the agent's performance.
- Visualize the results.

---

## Code Structure

### Key Components:
- **Environment Setup**: 
```
env = gym.make("CartPole-v1")
```
Initializes the CartPole environment.

- **Policy Network**: 
A neural network with one hidden layer that outputs action probabilities:
```
class Policy(nn.Module):
def init(self, s_size, a_size, h_size):
...
def forward(self, x):
...
def act(self, state):
...
```

- **Training Loop** (`reinforce`): 
Implements the REINFORCE algorithm to update policy weights based on rewards.

- **Evaluation**: 
Evaluates the trained agent on multiple episodes to measure performance.

- **Visualization**: 
Generates an animation of the agent's performance in the environment.

---

## Hyperparameters

The following hyperparameters are used in training:

| Parameter              | Value       |
|------------------------|-------------|
| Hidden Layer Size (`h_size`) | `16`        |
| Training Episodes (`n_training_episodes`) | `1000`      |
| Evaluation Episodes (`n_evaluation_episodes`) | `10`        |
| Maximum Timesteps (`max_t`) | `1000`      |
| Discount Factor (`gamma`) | `0.95`       |
| Learning Rate (`lr`) | `0.002`      |

You can modify these values in the `hyperparameters` dictionary in the notebook.

---

## Results

After training for 1000 episodes, the agent achieves an average reward of approximately **395.3** over 10 evaluation episodes, with a standard deviation of **111.71**.

Training progress is visualized as follows:


---

## Visualization

The trained agent's performance can be visualized using an animation:

```
images = record_video(env, policy)
HTML(animation.FuncAnimation(...).to_jshtml())
```

This will render an animation of the CartPole environment with the trained agent balancing the pole.

---

## Notes

- Deprecation warnings related to Gym's API may appear; these can be resolved by updating Gym and using `render_mode='rgb_array'` during environment initialization.
- Ensure all dependencies are installed before running the notebook.

Happy coding! 🎉

