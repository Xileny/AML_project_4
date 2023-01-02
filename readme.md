# Starting code for course project of Advanced Machine Learning (AML) 2022
"Sim-to-Real transfer of Reinforcement Learning policies in robotics" exam project.

**Note! This template has changed on Dec 21st, 2022. Make sure to clone the latest version.**


## Getting started

You can play around with the code on your local machine, and use Google Colab for training on GPUs. When dealing with simple multi-layer perceptrons (MLPs), you can even attempt training on your local machine.

Before starting to implement your own code, make sure to:
1. read and study the material provided
2. read the documentation of the main packages you will be using ([mujoco-py](https://github.com/openai/mujoco-py), [Gym](https://github.com/openai/gym), [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html))
3. play around with the code in the template to familiarize with all the tools. Especially with the `test_random_policy.py` script.


### 1. Local (Linux)

You can work on your local machine directly, at least for the first stages of the project. By doing so, you will also be able to render the Mujoco environments and visualize what's happening. This code has been tested on Linux with python 3.7 (Windows is somewhat deprecated, but may work also).

**Dependencies**
- Install MuJoCo and the Python Mujoco interface following the instructions here: https://github.com/openai/mujoco-py
- Run `pip install -r requirements.txt` to further install `Gym` and `Stable-baselines3`.

Check your installation by launching `python test_random_policy.py`.


### 1. Local (Windows)
As the latest version of `mujoco-py` is not compatible for Windows, you may:
- Try downloading a [previous version](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/) (not recommended)
- Try installing WSL2 (requires fewer resources) or a full Virtual Machine to run Linux on Windows. Then you can follow the instructions above for Linux.
- Stick to the Google Colab template (see below), which runs on the browser regardless of the operating system. This option, however, will not allow you to render the environment in an interactive window for debugging purposes.


### 2. Google Colab

You can also run the code on [Google Colab](https://colab.research.google.com/)

- Download all files contained in the `colab_template` folder in this repo.
- Load the `test_random_policy.ipynb` file on [https://colab.research.google.com/](colab) and follow the instructions on it

NOTE 1: rendering is currently **not** officially supported on Colab, making it hard to see the simulator in action. We recommend that each group manages to play around with the visual interface of the simulator at least once, to best understand what is going on with the Hopper environment.

NOTE 2: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.