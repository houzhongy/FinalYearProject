# This README file contains information about the custom codes for Final Year Project.
## Credits: https://github.com/cage-challenge/CybORG

### As the environment is developed by DST Group, all of the code written by me will be under CybORG/Agents/CustomCode.
Inside the folder, it contains a few different files.

#### 1. A2C.py
This python script contains the Advantage Actor Critic agent that I have implemented in this project.

#### 2. DeepQNetwork.py
This python script contains the Deep Q-Network agent that I have implemented in this project

#### 3. cartpole_script.ipynb
This jupyter notebook file contains the code implementation of my Reinforcement Learning agent on GYM's Cartpole-v1 environment.

#### 4. CustomWrapper.py
This python script contains the implementation of the custom wrapper that was being used to reduce the observation space of the environment.

#### 5. Main.ipynb
This jupyter notebook is the main file that contains all the code with regards to CybORG environment. All of the implementations with regards to CybORG environment can be found in this file.


* If you would like to run the files, please ensure that a python version of < python 3.8 are used. If not, there will be depreciation issues and the code might not run as determined.
This project runs on python 3.7.16 and pytorch 1.13.1. Do note that pytorch CUDA no longer support python 3.7, thus this project experiments are all running on CPU resources.