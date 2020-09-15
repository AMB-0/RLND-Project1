# Project 1 - Banana collector agent

The current Report.md file summarizes the implementation, algorithm used and the results obtained in the Project 1 of the Udacity's Deep Reinforcement Learning Nanodegree. For a brief description of the project, environment and how to get started, please check the <a href="../README.md">README.md</a> file.  This document is structured as follows:

<br/>

## Table of Contents
1. [Framing the problem](#Framing-the-problem)
2. [Deep Q-Learning](#Deep-Q-Learning)
    1. [Key Ideas](#Key-ideas)
    2. [How it works](#How-it-works)
3. [Implementation & results](#Implementation-&-Results)
    1. [Code Structure](#Code-Structure)
    2. [Neural Network Architecture](#Neural-Network-Architecture)
    3. [Hyperparameters](#Hyperparameters)
    4. [Results](#Results)
4. [Future work](#Future-Work)

<br/><br/>

# Framing-the-problem

As described in the <a href="../README.md">README.md</a> file, the goal of this project is to train an agent to navigate a large square virtual world while collecting bananas. The caveat here is that yellow bananas deliver a positive reward of +1 while the blue bananas returns a negative reward of -1. Hence the idea is that the agent navigates while picking up as much yellow bananas as possible while avoiding the blue ones. The agent can achieve this by choosing the right action - move either left, right, forward or back - at each timestep. However, part of the goal is that the agent can learn how to accomplish this goal without explicit instructions but through reinforcement learning.
<br/><br/>
We can think of this problem as each timestep being a sort of snapshot of the environment in which the agent takes an action and receives a reward being either 0, +1 or -1. Following this idea, we can then frame this situation as a Markov Decision Processes as shown in the following diagram:

![figure1]

- The state is given by the position of the agent, it's velocity and the relative position of the objects in the virtual world
- The actions are essentially four, move left, right, forward or back
- The rewards are given by the bananas picked up by the agent

<br/>

The key idea behind how we can solve this type of problems is the "experience". Essentially we can train the agent to understand how to navigate the virtual world just by learning of its own interactions with the environment without any previous knowledge. Hence, we can start with an agent that chooses random actions at each state and, by keeping track of the outcomes / rewards obtained by each action, the agent can then "learn" which action to take in the future to maximize its reward. 

<br/>

We can keep track of the rewards for each action-state pair in a table, usually called Q-table or action-value function table and then pick the action with the maximum value for each state. This works fine as long as the number of states / actions is finite. However, in problems like the one described before, where the number of states / actions is infinite, the number of entries in the Q-table would be infinite as well which makes this solution not feasible. 

<br/>

One way of approaching problems with either continuos state space or action space is to use neural networks as a "black box" for estimating this action-value function. In short, we can use a neural network as a replacement for the Q-table and adjust its weights to estimate the action-value function. We can then choose the action with the maximum output as the best action for each given state. In the next section, we will describe and implement one algorithm called Deep Q-Learning or DQN that follows this idea to solve this challenge. 

<br/><br/>

# Deep-Q-Learning

The Deep-Q learning algorithm, introduced by Mnih. et al in the article called <a href=https://daiwk.github.io/assets/dqn.pdf>Human-level control through deep reinforcement learning</a>, draws on the following key ideas for solving MDP problems like the one in this project:

## Key-Ideas

1. <u>Online sampling / Offline training with replay buffer</u><br/>
Reinforcement learning algorithms like Sarsa or Q-learning use online training. This means that each time the agent interacts with the environment, by choosing an action, the agent will learn from that outcome and then discard the information. This is not optimal since some states are less likely to be experienced and we might want to learn from them several times. In addition, there's a clear correlation in how states are presented to the agent since some states might only appear after several decisions were made. By learning following this same sequence, we are introducing a bias into our agents' learning pattern.<br/>
DQN solves this issue by using online sampling and offline training. This means that, instead of learning while we interact with the environment, the agent saves its experiences in a replay buffer. The agent then draws random samples mini-batches from this buffer in an "offline" setting and learn from them later on.
<br/>
<br/>
2. <u>Exploration v/s Explotation (e-greedy policy)</u><br/>
A second key element on reinforcement learning algorithms is how to choose the action in each time step. In the case of DQN algorithm, the agents' action is selected by following an e-greedy policy. This means it depends on a parameter "epsilon" that goes between 0 and 1 as follows:
    - With probability "1-epsilon" --> We select the action that has maximum the value in the Q-table / action-value function for the given state
    - With probability "epsilon" --> We select a random action from the possible action space
<br/>

    The "e-greedy" policy described, ensures that the agent explote the best action while keeping exploration of the rewards for some other actions, hence, avoiding getting stuck into a suboptimal policy.
<br/>
<br/>
3. <u>Detached learning from training</u><br/>
Another important setting from DQN algorithm is the use of two neural networks instead of just one. This helps detach the learning step from the training step. The issue that we want to avoid here is to update the weights of the neural network - with the learning experiences drawn from the environment - and then used this same neural network for estimating the target for the next sample. This would cause the loss to be very noisy, hence impacting the algorithm stability. An analogy for this could be to try to aim for a moving target. By having two different neural networks, we can use one from learning and another one from estimating the target "y_i" which effectively detached the learning from the training step.

4. <u>Soft update</u><br/>
One final addition to DQN is to include a "Tau" parameter for controlling how the weights from the Q_target network are updated using the Q_local network. This "Tau" parameter take values between 0 and 1 and work as follows:
    - A Tau value of 1 means that we overwrite all weights of Q_target with the Q_local weights
    - A Tau value of 0 means that we don't update the weights of Q_target at all
    - A Tau value of 0.3 means that we update the weights of Q_target by considering a 30% of the value of Q_local and a 70% of the value of Q_target
<br/><br/>
## How-It-Works

In a nutshell, the DQN algorithm works by first initializing the values / parameters and then applying a 2-phase process - a sampling and a learning phase - as follows:

0. <u>Initializing phase</u>:<br/>
The algorithm starts by initializing all the variables / parameters required. This includes creating an empty replay buffer, creating the two neural networks - one for the local value and one for the target value - in addition to their respective weights.  

1. <u>Sampling phase</u>:<br/>
In this phase the agent interacts with the environment by choosing its actions based on the e-greedy policy. Then, the current state, the action performed, the reward received and the next states are stored in the replay buffer as a tuple. This information will be used later-on for adjusting the neural networks weights.

2. <u>Learning phase</u><br/>
In this phase the agent starts by drawing a random minibatch of experiences from the replay buffer. Each experience of the minibatch is then used to estimate a target value "y_i" as follows:
    - "y_i" = reward_i                               ; if episodes terminates at next_state_i
    - "y_i" = reward_i + Q_target(next_state_i, action_i)   ; otherwise<br/>

    This "y_i" target is then used to estimate the MSE (mean square error) loss versus the Q_local(state_i, action_i) value. The weights of the Q_local network are then adjusted using gradient descent to minimize the loss. It's important to note here how the two neural networks are not in sync and how both are used for different purposes. In one hand, the Q_target network is used as a fixed base point and we use it for estimating the "y_i" target while the Q_local is continuosly being updated with the experiences drawn from the replay buffer.
<br/>
<br/>
After a fix number of steps, we update the weights of Q_target with Q_local using the Tau parameter described above. 


<br/>

# Implementation-&-Results

## Code-Structure

For the implementation of the DQN algorithm, 3 different files were used. These files are included within the src folder of the project as shown below:


```
PROJECT 1
│
│ README.md
│
├───docs
│       Report.md
│
├───img
│       Results.png
│
└───src
        Agent.py
        checkpoint.pth
        model.py
        Navigation.ipynb
```

<br/>

Description of files:

1. <u><a href=../src/model.py>Model.py</a></u>:<br>
    File that contains the implementation of the neural network. A description of this architecture can be found in the following section.
    <br><br>
2. <u><a href=../src/Agent.py>Agent.py</a></u>:<br>
    File that contains the implementation of the agent class and the respective replay buffer class
<br><br>
3. <u><a href=../src/Navigation.ipynb>Navigation.ipynb</a></u>:<br>
    Jupyter notebook including all the necessary steps for training the agent from scratch. Even though the task is considered with a average reward of 13 over the last 100 episodes, the code stops if an average reward over 15 is achieved
<br><br>

## Neural-Network-Architecture
<br>
The architecture used for the Q-table neural network is quite simple. It receives the current state of the environment as the input and it returns the action-value function of the agent. The architecture includes 3 fully connected layers with the following characteristics:<br>
        - <u>FC layer 1</u>: Layer with 64 nodes fully connected with FC layer 2. This layer receives "N" inputs where "N" corresponds to the number of dimensions in the state space (37 in this case). The activation function for this layer is a rectifier layer unit (ReLU)<br>
        - <u>FC layer 2</u>: Layer that also has 64 units that fully connected with FC layer 3. The activation function is also a ReLU<br>
        - <u>FC layer 3</u>: Layer that receives 64 inputs coming from FC layer 2 and outputs "M" values, one for each action<br>

![figure2]
<br><br>

## Hyperparameters
The value for each hyperparameters used as part of the implementation is included within the <a href=../src/Agent.py>Agent.py</a> code
<br><br>

## Results

The results obtained are summarized as follows:

![figure3]

By looking at the plot above, we can see that the agent started with a reward of 0 and while it was exploring the different state/action pair it was improving it's reward. After a couple of hundred episodes, the agent's learning curve started to increase until it finally reached an average reward of 15 after 532 episodes. <br><br>
The weights obtained as part of the training are included in the <a href="../src/checkpoint.pth">checkpoint.pth</a> file.
<br><br>

# Future-Work

A couple of ideas to try in the future:

- Add some variations to the DQN algorithm:
    - Double DQN
    - Use a prioritized Replay Memory, adding more weight to some learning experiences
    - Dueling Networks Architecture

- Try some policy-based methods algorithms
- Try adding more complexity to the neural network architecture  


[figure1]: https://video.udacity-data.com/topher/2017/September/59c29f47_screen-shot-2017-09-20-at-12.02.06-pm/screen-shot-2017-09-20-at-12.02.06-pm.png "MDP diagram"
[figure2]: ../img/NN_diagram.png "Neuronal Network Diagram"
[figure3]: ../img/Results.png "DQN Results"