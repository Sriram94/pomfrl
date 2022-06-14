# Partially Observable Mean Field  Reinforcement Learning 

Implementation of POMFQ for the AAMAS-2021 paper Partially Observable Mean Field Reinforcement Learning. The paper can be found [here](https://arxiv.org/abs/2012.15791).


The environments contain 2 teams training and fighting against each other. 
 
## Code structure

- See folder pomfrlFOR for training and testing scripts of the FOR environment. 

- See folder pomfrlPDO for training and testing scripts of the PDO environment. 

- See folder isingmodel for the implementation of POMFQ with the ising model. 

### In each of directories, the files most relevant to our research are:

- /pomfrlFOR/examples/battle_model/python/magent/builtin/config: This folder contains the reward function for all the games. It is same for the 2 environments. 

- /pomfrlFOR/examples/battle_model/senario_battle.py: Script to run the training and testing for the battle game. This has the action aggreagation calculation for all the algorithms.

- /pomfrlFOR/train_battle.py: Script to begin training the Multibattle game for 2000 iterations. The algorithm can be specified as a parameter (MFAC, MFQ, IL, or POMFQ). You can also run the recurrent algorithms using rnnIL or rnnMFQ as the parameter. Similarly you can run the train_gather and the train_pursuit.py files for the Battle-Gathering and the Predator-Prey domains. These scripts were used run the training experiments.

- /pomfrlFOR/battle.py: Script to run comparative testing in the Multibattle game. The algorithm needs to be specified as a command line parameter. All the algorithms described in the previous point can be used as the parameter. Similarly the gather.py and pursuit.py files run the test experiments for the Battle-Gathering and Predator-Prey domains.  These scripts were used to get all the test (faceoff) experiments. 

- /pomfrlFOR/examples/battle_model/algo: This directory contains the learning algorithms.

- /pomfrlFOR/examples/battle_model/python/magent/gridworld.py: This script contains the code changes for the modified partially observable games used in our paper compared to the previous MAgent games. 

All of the above pointers also holds for the PDO domain. Just look at the relavant files in the pomfrlPDO folder. 

- /isingmodel/main_POMFQ_ising.py: This file is used to run the POMFQ algorithm with the ising model.

## Installation Instructions for Ubuntu 18.04

### Requirements

Atleast 

- `python==3.6.1`


```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```

- `gym==0.9.2`


```shell
pip install gym
```

- `scikit-learn==0.22.0`


```shell
sudo pip install scikit-learn
```


- `tensorflow 2`

```shell
pip install --upgrade pip
pip install tensorflow
```


- `libboost libraries`


```shell
sudo apt-get install cmake libboost-system-dev libjsoncpp-dev libwebsocketpp-dev
```
 


Download the files and store them in a separate directory to build the MAgent framework. 

#### Build the MAgent framework 

```shell
cd /pomfrlFOR/examples/battle_model
./build.sh
```

Similarly change directory and build for the PDO domain. 

### Training

```shell
cd pomfrlFOR
export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
python3 train_battle.py --algo il
```

This will run the training for the multibattle domain with the IL algorithm. For other algorithms, specify MFQ, MFAC or POMFQ in the --algo command line argument. Change directory to pomfrlPDO to run the PDO experiments. 
 

### Testing

```shell
cd pomfrlFOR
python battle.py --algo il --oppo mfq --idx 1999 1999
```

The above command is for running the test battles in the FOR setting. You need to specify the algorithms as command line arguments and give the path to the correct trained model files within this script.  

When running test battles with POMFQ, you need to additionally specify the position of POMFQ as a command line parameter. 

```shell
cd pomfrlFOR
python battle.py --algo il --oppo pomfq --idx 1999 1999 --pomfq_position 1
```


Similarly, change directory to pomfrlPDO, for PDO experiments.

Repeat all the above instructions to train and test the other two games. 

train\_gather.py and gather.py runs the train and test for the Battle-Gathering domain and train\_pursuit.py and pursuit.py runs the train and test for the Predator-Prey domain. 
 



For more help with the installation, look at the instrctions in [MAgent](https://github.com/geek-ai/MAgent), [MFRL](https://github.com/mlii/mfrl) or [MTMFRL](https://github.com/BorealisAI/mtmfrl). 
In these repsitories installation instructions for OSX is also provided. We have not tested our scripts in OSX. 

To run the Ising model 


```shell
cd isingmodel
python main_POMFQ_ising.py 
```


## Code citations 

We would like to cite the [MAgent](https://github.com/geek-ai/MAgent) for the source files for the three game domains used in this paper. We have modified these domains to be partially observable as described in the code structure. We would like to cite [MFRL](https://github.com/mlii/mfrl) for the source code of MFQ, MFAC and IL used as baselines and also for the Ising model environment. Both these repositories are under the MIT license. 


## Note

This is research code and will not be actively maintained. Please send an email to ***s2ganapa@uwaterloo.ca*** for questions or comments. 



## Paper citation

If you found this helpful, please cite the following paper:

<pre>



@InProceedings{Srirampomfrl2021,
  title = 	 {Partially Observable Mean Field Reinforcement Learning},
  author = 	 {Subramanian, Sriram Ganapathi and Taylor, Matthew E. and Crowley, Mark and Poupart, Pascal} 
  booktitle = 	 {Proceedings of the International Conference on Autonomous Agents and Multi Agent Systems (AAMAS 2021)},
  year = 	 {2021},
  editor = 	 {U. Endriss, A. Nowé, F. Dignum, A. Lomuscio},
  address = 	 {London, United Kingdom},
  month = 	 {3--7 May},
  publisher = 	 {IFAAMAS}
}
</pre>




