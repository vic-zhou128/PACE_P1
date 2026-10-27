# Unitree RL Lab

P1 的环境配置已统一到 [p_pace/velocity_env_cfg.py](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/p_pace/velocity_env_cfg.py)。
文件前半部分完整定义当前 **TrueLiftSmooth** 任务的场景、事件、命令、动作、观测、奖励、终止条件、课程以及训练/播放环境；历史任务的差异集中在文件末尾。
当前调参请直接修改该文件的 `RewardsCfg`、`CommandsCfg`、`ActionsCfg` 等类，配置入口见 [P_TRUE_LIFT_SMOOTH.md](P_TRUE_LIFT_SMOOTH.md)。

本目录新增了 P 机器人任务 **`Unitree-P-G1Rewards-PACE`**：使用宇树官方 G1 奖励与 PPO、`robot-p` 中带步态相位的 P 课程，以及 PACE 识别后的电机模型。训练入口、参数来源和验证结果见 [P_TRAINING.md](P_TRAINING.md)。
针对站立局部最优，新增行走任务 **`Unitree-P-G1Rewards-PACE-Walk`**，其奖励修正、训练与定速评估见 [P_WALKING_V2.md](P_WALKING_V2.md)。
按宇树 G1 域随机化另设行走对照任务 **`Unitree-P-G1Rewards-PACE-Walk-G1Events`**，训练命令和差异见 [G1_EVENTS_WALK.md](G1_EVENTS_WALK.md)。

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

  - Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Config `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files



- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l # This is a faster version than isaaclab
    ```
  - Running a task:

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - Inference with a trained agent:

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    ```

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name.
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
