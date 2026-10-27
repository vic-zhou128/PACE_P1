# P 机器人：宇树 G1 域随机化行走训练

新任务：`Unitree-P-G1Rewards-PACE-Walk-G1Events`。它继承仓库内固定版本的宇树 G1 环境、17 项适用于 P 的奖励及 PPO 配置；P 的 12 关节、PACE 电机、0.8 s 步态相位、原 P 逐轴速度课程和行走优先指令保持不变。PPO 网络与优化器沿用 G1 基础配置；行走版将初始动作噪声设为 0.35、熵系数设为 0。为防止静止足端得到抬脚奖励，仍保留行走版对抬脚奖励的修正，并将前进速度跟踪权重设为 2.0、误差尺度设为 0.20 m/s。原先的训练入口和 checkpoint 均不受影响。

事件直接继承宇树 G1 的 `EventCfg`，仅把 `torso_link` 适配为 P 的 `pelvis_link`：

| 事件 | 生效设置 |
| --- | --- |
| 摩擦 | 静/动摩擦均为 0.3–1.0，启动时采样 |
| 机身质量 | pelvis_link 增加 −1.0 至 +3.0 kg，启动时采样 |
| 外力/力矩 | G1 默认均为 0 |
| 复位 | x/y 各 ±0.5 m，yaw ±3.14 rad；关节位置默认值，关节速度 ±1.0 |
| 推扰动 | 每 5 s，x/y 速度各 ±0.5 m/s |
| 电机 | PACE 识别参数固定，不额外随机化 |

原 P 的推扰动课程会在复位时覆盖 G1 的推扰动幅度，因此**此新任务只保留原 P 的逐轴速度课程**；旧 `Unitree-P-G1Rewards-PACE-Walk` 任务仍可用于完整 P 速度与推扰动课程的对照。训练场景继续用 P 的平地与接触传感器；G1 的高度扫描观测在官方配置中本来就被注释。

从头训练：

```bash
conda activate isaaclab
cd /home/pc/PACE_P1/unitree-p-pace
python scripts/rsl_rl/train.py --headless \
  --task Unitree-P-G1Rewards-PACE-Walk-G1Events \
  --num_envs 4096 --max_iterations 3000 --seed 42 \
  --run_name g1_events_walk_from_scratch
```

训练日志：`logs/rsl_rl/unitree_p_g1_pace_walk_g1_events/`。可视化时去掉 `--headless`，并提供该新任务目录内的 checkpoint；第 1–2 轮的烟雾测试 checkpoint 不代表行走策略。

已用 8 个环境完成 2 轮 PPO 烟雾测试，环境与奖励成功初始化，保存的 `params/env.yaml` 含上述 G1 事件及 P 速度课程；还完成了 3 步无界面回放入口测试。尚未运行完整 3000 轮训练，收敛效果需看新日志。
