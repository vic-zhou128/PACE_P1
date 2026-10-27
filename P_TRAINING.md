# P 机器人：官方 G1 奖励、原 P 课程、PACE 电机

此文档记录第一版训练配置。行走改进版及其奖励修正请看 [P_WALKING_V2.md](P_WALKING_V2.md)。

项目目录：`/home/pc/PACE_P1/unitree-p-pace`；训练任务：`Unitree-P-G1Rewards-PACE`。

基于 [宇树官方 unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)，版本固定为 `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`。官方 G1 奖励配置、奖励函数文件、基础 PPO 配置保持原样，新增 P 任务通过继承和适配使用它们。

## 开始训练

当前机器已经有可用的 `isaaclab` 环境。训练与回放脚本优先导入本目录的源码，不需要重新安装其他目录的 `unitree_rl_lab`。

```bash
conda activate isaaclab
cd /home/pc/PACE_P1/unitree-p-pace
python scripts/rsl_rl/train.py --headless \
  --task Unitree-P-G1Rewards-PACE --num_envs 4096 \
  --run_name p_g1_rewards_pace
```

默认训练上限为官方配置的 50000 轮；可用 `--max_iterations` 修改。日志、模型和部署参数保存在本目录的 `logs/rsl_rl/unitree_p_g1_rewards_pace/<时间戳>_<名称>/`。

训练完成后，在有图形显示的会话中回放指定模型：

```bash
python scripts/rsl_rl/play.py --task Unitree-P-G1Rewards-PACE \
  --num_envs 1 --checkpoint /绝对路径/model_XXXX.pt
```

回放使用完整速度范围，关闭课程、推扰动以及质量、质心、地面摩擦随机化；保留 PACE 电机模型。SSH 会话若没有图形显示，可使用已有的远程图形会话，或加入 `--headless` 运行。

本任务需要从头训练。观测为 actor 235 维、critic 250 维，动作 12 维；旧 P 策略的观测结构不同，不能直接续训。两轮验证产生的 `smoke_g1_rewards_pace` 模型仅用于检查训练链路。

## 奖励与训练框架

奖励直接继承 [官方 G1 RewardsCfg](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py)。以下 17 项的函数与权重保持官方设置：

| 奖励项 | 权重 |
| --- | ---: |
| track_lin_vel_xy | 1.0 |
| track_ang_vel_z | 0.5 |
| alive | 0.15 |
| base_linear_velocity | -2.0 |
| base_angular_velocity | -0.05 |
| joint_vel | -0.001 |
| joint_acc | -2.5e-7 |
| action_rate | -0.05 |
| dof_pos_limits | -5.0 |
| energy | -2e-5 |
| joint_deviation_legs | -1.0 |
| flat_orientation_l2 | -5.0 |
| base_height | -10.0 |
| gait | 0.5 |
| feet_slide | -0.2 |
| feet_clearance | 1.0 |
| undesired_contacts | -1.0 |

为适应 P 的结构，禁用了不存在的手臂、腰部关节偏离项；髋关节和足端名称改为 P 的名称；目标根部高度从 G1 的 0.78 m 改为 P 的 0.656 m。足端显式按左、右排序，避免相位错位。

步态奖励维持官方周期 0.8 s、左右相位差 0.5、站立比例阈值 0.55；足端高度奖励目标维持 0.1 m。这里的足端高度是 ankle_roll 刚体原点高度。actor 与 critic 都加入同周期的正余弦步态相位，历史长度为官方的 5 帧。零速度指令不会激活官方步态奖励。

训练使用官方 `OnPolicyRunner`、高斯 `ActorCritic` 与 PPO。网络尺寸为 `[512, 256, 128]`，每环境每轮采样 24 步，初始学习率 `1e-3`。针对 P 的有界位置动作，包装器设置 `clip_actions=1.0`，并沿用原 P 的各关节动作尺度和位置偏移限幅。

## 原 P 课程与随机化

课程来源是 `robot-p/.../tasks/locomotion/robots/p/velocity_env_cfg.py` 和对应 `mdp/curriculums.py`，使用带相位的 **p** 配置。两项课程及其参数已复制到新项目，后续修改原项目不会悄悄改变新任务。

| 轴 | 初始指令范围 | 最终范围 | 每次晋级扩展量 |
| --- | --- | --- | --- |
| 前后速度，m/s | [-0.12, 0.18] | [-0.45, 0.55] | 0.03 |
| 横向速度，m/s | [-0.08, 0.08] | [-0.30, 0.30] | 0.02 |
| 转向速度，rad/s | [-0.15, 0.15] | [-0.60, 0.60] | 0.04 |

`velocity_levels` 每 10000 个策略步检查一次，保留原课程的最小样本要求、逐轴速度跟踪评分、站立评分、生存率及晋级/回退阈值。`push_levels` 随速度课程晋级增强推扰动，回退时同步减弱：x/y/yaw 范围由 `(0.04, 0.03, 0.06)` 增长至 `(0.18, 0.12, 0.18)`。

保留原 P 的指令混合采样、速度变化速率、25% 站立指令、地面摩擦、质量、质心、复位随机化与终止条件。识别后的 PD、armature 和摩擦参数固定，关闭原电机增益和关节参数随机化。动作推理延迟、通信延迟及电机执行器延迟均为 0。

站立评分仍参与原课程考核；奖励采用上述官方 G1 项，原 P 的额外站立位置锚定奖励没有加入。本次只验证课程逻辑与训练运行，行走收敛和课程实际晋级速度需通过正式训练评估。

## PACE 参数与 URDF

训练默认读取仓库内 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/data/p1_parameters_old_vs_new.csv` 的“新版”列，路径相对于参数加载模块解析，克隆到其他电脑后无需额外放置父目录 CSV。该文件已纳入 Git，且与原 `/home/pc/PACE_P1/p1_parameters_old_vs_new.csv` 的参数一致。加载时检查名称、单位、重复项、有限数值及 12 个关节参数的完整性。可以通过 `P_PACE_PARAMETERS_CSV` 指定另一个 CSV。

URDF 已对齐为 `/home/pc/unitree_usd/urdf_803_nobattery/urdf/p1.urdf`。可通过 `P_ROBOT_URDF` 指定其他文件；任务初始化会核对其 12 个活动关节与识别参数映射。

每个关节使用 CSV 中的 armature、黏性摩擦、静/动摩擦系数和编码器 bias。静摩擦和动摩擦系数相同，黏性摩擦由 PhysX 计算，PD 为所有关节 `kp=180, kd=10`。bias 按 `q_sim=q_encoder+bias` 处理，复位姿态、动作零位、PD 反馈和部署导出使用一致的坐标关系。

力矩—速度曲线取自 `pace-sim2real/source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/p1_pace_env_cfg.py`：

| 关节组 | 堵转/峰值力矩，Nm | 输出力矩上限，Nm | 电机空载速度，rad/s |
| --- | ---: | ---: | ---: |
| hip_roll / hip_pitch | 281.6 | 120 | 16.57 |
| hip_yaw / knee_pitch | 110.25 | 60 | 20.56 |
| ankle_pitch / ankle_roll | 181 | 68 | 8.75 |

使用与 PACE 相同的 Isaac Lab DC 电机限幅曲线；电机空载速度参与力矩—速度限幅，PhysX 的关节硬速度上限保留 URDF 设置。踝关节输出上限为 PACE 配置中的 68 Nm。

CSV 识别延迟为 1.483564853668213 个 1000 Hz 仿真步，按识别实现截断后为 1 步；依照你的要求，训练执行器延迟设置为 **0**。训练物理步长保持官方和原 P 的 0.005 s，策略步长 0.02 s（50 Hz）。若后续要研究识别与训练积分步长的差异，需要另行对比 1000 Hz 仿真。

PACE 源配置说明 ankle_pitch 使用对应 ankle_roll 的拟合值；因此这里沿用这些值，不把 ankle_pitch 描述为单独辨识的结果。

部署导出 `params/deploy.yaml` 包含所用 CSV 路径及 SHA-256、每关节识别值、PD、曲线、实际延迟、物理/编码器零位和关节顺序。该文件记录训练控制参数；本次未适配 P 的真实机器人通信与部署程序。

## 文件位置与检查结果

| 内容 | 新项目内路径 |
| --- | --- |
| P 任务与 G1 奖励适配 | `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/p_pace/velocity_env_cfg.py` |
| 当前环境与历史任务配置 | 同一 `velocity_env_cfg.py`；历史差异集中在文件末尾 |
| P 课程和动作适配函数 | `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/p_robot/` |
| URDF、初始化姿态、电机配置 | `source/unitree_rl_lab/unitree_rl_lab/assets/robots/p_pace.py` |
| CSV 加载与力矩—速度常量 | 同目录 `p_pace_parameters.py` |
| PACE DC 电机实现 | 同目录 `pace_actuator.py` |
| 官方版本与复制来源记录 | `P_TRAINING_SOURCES.json` |

已在当前机器的 Isaac Sim 5.1、Isaac Lab 2.3.2、rsl-rl-lib 3.1.2、RTX 4090 上完成：

- 官方 G1 配置、奖励函数和基础 PPO 文件与克隆版本逐字节比较；17 个有效奖励的函数和权重比较。
- 原 P 课程函数与参数比较；模拟充足样本验证逐轴晋级、回退、同步推扰动以及重复调用的行为。
- 12 关节 CSV 参数与 PACE 源配置比较；三类电机在正反转、制动、空载速度附近、超速、0/1 步延迟下与 PACE 原实现的输出比较。
- 4 个实际仿真环境运行 12 个策略步，验证 PhysX 参数、bias、延迟、有限观测/奖励、重置与部署导出。
- 官方训练入口以 64 个环境完成 2 轮 PPO、3072 个采样步，正常退出，保存 `model_0.pt`、`model_1.pt` 与训练配置。

重新运行参数与环境检查：

```bash
python scripts/check_p_training.py --headless --device cuda:0
```

短训练验证日志为 `/tmp/unitree_p_g1_rewards_ppo_check.log`，参数与环境检查日志为 `/tmp/unitree_p_g1_rewards_check.log`。没有启动正式长程训练；上述检查证明训练链路可运行，不代表策略已学会稳定行走。
