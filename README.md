# Project 1 - Banana collector agent

The current README.md file summarizes the project, environment, goals and how to get started with the Project 1 of the Udacity's Deep Reinforcement Learning Nanodegree. The structure and content of this document is included in the table of contents below. 

<br/>

A summary of the implementation / solution of the current project can be found in the <a href="../docs/Report.md">Report.md</a> file.

<br/>

## Table of Contents
1. [Project description](#Project-Description)
2. [Environment](#Environment)
3. [Getting Started](#Getting-Started)

<br/>
<br/>


# Project-Description

The project consists on training an agent to navigate a large square virtual world collecting bananas. For each banana the agent collects it will receive a reward: the yellow ones delivering a reward of +1 while the blue one a reward of -1. Thus, we are only interested in collecting the yellow bananas while avoiding the blue ones. 

<br/>

The goal of the project if to maximize the agent's reward and will be considered solve if the agent scores more than +13 over 100 consecutive episodes. It's important to note that each episode will last for 300 steps.

<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![alt text][figure1]

[figure1]: https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif "Agent environment"

<br/>

This project is actually a simplified version of one of the learning environments of Unity ML-Agents platform which can be found in the <a href=https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector>Unity ML-Agents GitHub page.</a>

<br/>

## Environment

<br/>

The agent-environment interaction happens as follows: at each time step, the environment sends its current state to the agent which contains 37 dimensions describing the following:

- Agent's velocity in the plane (2 entries)
- 7 ray-based elements that describe the information regarding the surroundings of the agent. These rays shows potential objects in the agent's way in its forward direction for the following angles 20°, 45°, 70°, 90°, 110°, 135° and 160°. 

Also, out of the 5 entries in each ray, the first 4 entries corresponds to a one-hot encoding of the type of object the ray might hit in the following order: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[yellow banana, purple banana, wall, nothing]

The last entry consist of the percent of the ray length at which the object was found.

<br/>

![alt text][figure2]
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 1: Agent's world diagram [1]

[figure2]: https://wpumacay.github.io/research_blog/imgs/img_banana_env_observations.png "Agent environment"

<br/>

To clarify, in the example (a) in the previous figure, the first row corresponds to the data sent by the ray that forms a 20° angle with the agent's plane and is showing [0, 1, 0, 0, 0.17910448] which means that there is a purple banana in this direction at a distance corresponding to 17910448 of the total length of the ray.

<br/>

Now, given this information, the agent has to learn, and of course choose the best action for the current state from four possible options available:

0. move forward
1. move backward
2. turn left
3. turn right

<br/><br/>

# Getting-Started

The content of the project within this repository is organized as follows:

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

The project was built using python 3.6, pytorch and some other libraries included within the requirements.txt file. <br/>

For running the project the followings steps are required:

1. Create an environment with python 3.6. For this, the following steps are recommended<br/>
        1.1.  install miniconda (link: https://docs.conda.io/en/latest/miniconda.html)<br/>
        1.2.  run the following on the command prompt:<br/>
        conda install python=3.6<br/>
        conda create -n rlnd-env python=3.6
2. Clone the project / Download the files within the project
3. Install the required python libraries by running the following on the command prompt:<br/>
        python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
4. Download unity agent file following one of the following links depending on the operating system:<br/>
        - Linux: <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip>click here</a><br/>
        - Mac OSX: <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip>click here</a><br/>
        - Windows (32-bit): <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip>click here</a><br/>
        - Windows (64-bit): <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip>click here</a><br/>
Then, place the file in the same folder as the project folder and unzip (or decompress) the file.
5. Open the "navigation.ipynb" file and follow the instructions for modifying the "file_name" parameter based on the unity agent file downloaded in step 4
6. Run the code in the "navigation.ipynb" file


## Reference
[1] https://wpumacay.github.io/research_blog/posts/deeprlnd-project1-part1/
