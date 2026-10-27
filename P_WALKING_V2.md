# P 行走任务：摆脱站立局部最优

新任务：`Unitree-P-G1Rewards-PACE-Walk`。原任务 `Unitree-P-G1Rewards-PACE` 及其 50,000 轮 checkpoint 保留，便于对照。新任务需要单独训练；旧 checkpoint 使用的是不同的奖励与高噪声参数，不能直接作为新任务的正常训练结果。

## 为什么旧版会站着拿高分

在这台机器完成的旧版训练中，50,000 轮后速度课程仍停在初始范围，前后/横向/转向跟踪分数约为 0.56/0.74/0.60，动作噪声标准差达 146，远超 ±1 的动作限幅。起始前进指令最高仅 0.18 m/s，还有 25% 站立指令。

[官方 G1 `foot_clearance_reward`](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py) 计算 `exp(-高度误差² × 足端速度因子 / std)`；足端速度为零时，指数中的乘积为零，所以静止足端始终获得 1 分。旧版日志里该项末值约 0.92，即使可视化显示机器人基本在站立。原官方步态奖励在双脚一直着地时也能匹配站立相位的一部分。

## 新任务的训练选择

- 保留同一份 PACE CSV、URDF、每关节 armature/摩擦/bias、PD `180/10`、力矩—速度曲线和零执行器延迟。
- 保留原 P 的逐轴速度课程、同步推扰动课程、外部域随机化、相位观测与其余适用于 P 的官方 G1 奖励。
- 将 `feet_clearance` 改为仅在**有运动指令、对应摆动相、足端离地且水平运动**时给奖励。高度仍使用官方目标 0.1 m、参数 `std=0.05` 和速度系数 2.0；静止足端得零分。这是对官方 G1 奖励的一项明确修正。
- 前进速度跟踪仍使用官方同一函数，但 P 的权重设为 2.0、误差尺度设为 0.20 m/s，使跟随 0.25 m/s 指令明显优于慢移。其他奖励权重保持旧任务设置。
- 初始前进范围改为 0.18–0.30 m/s，命令模式按前进/横移/转向/混合 70%/10%/10%/10% 采样，站立比例降为 5%。仍使用原 P 的逐轴课程函数；行走版把站立分数从晋级条件中移除，生存率晋级/回退门槛设为 0.85/0.70，跟踪分数门槛仍为 0.75。横移和转向只有各自达标才扩展。
- 使用同一 PPO/MLP 架构，从头训练，初始高斯噪声设为 0.35，采用对数标准差，熵系数设为 0。短训练中噪声由 0.35 降至约 0.29。

训练入口：

```bash
conda activate isaaclab
cd /home/pc/PACE_P1/unitree-p-pace
python scripts/check_p_walk.py --headless
python scripts/rsl_rl/train.py --headless \
  --task Unitree-P-G1Rewards-PACE-Walk --num_envs 4096 \
  --run_name walking_first
```

新日志在 `logs/rsl_rl/unitree_p_g1_rewards_pace_walk/`；旧实验目录不受影响。固定前进回放用 `WalkPlayEnvCfg` 的 0.25 m/s、零横移和零转向指令，适合检验是否真正向前走：

```bash
python scripts/rsl_rl/play.py --task Unitree-P-G1Rewards-PACE-Walk \
  --num_envs 1 --real-time --checkpoint /绝对路径/model_XXXX.pt
```

SSH 无图形桌面时可加 `--headless --video --video_length 1000` 生成约 20 秒 MP4。量化实测速度：

```bash
python scripts/eval_p_walk.py --headless --steps 1000 \
  --checkpoint /绝对路径/model_XXXX.pt
```

检查脚本验证了静止、着地、零指令足端都不能获得抬脚奖励，并在 4 个真实 Isaac Sim 环境中步进 32 次。相同 256 环境、300 轮诊断训练后，使用固定 0.25 m/s 指令评估：修正抬脚奖励后实测 0.063 m/s；进一步收紧前进跟踪奖励后实测 **0.168 m/s**，两次评估均运行 1000 步且没有重置。继续从较好的诊断模型训练到第 998 轮后，同一固定指令的实测速度为 **0.291 m/s**、平均绝对误差 **0.041 m/s**。16 个环境各运行 1000 步，零重置。再续训到第 1497 轮后，前进跟踪评分约 0.900、生存率约 0.967，原 P 逐轴课程将前进范围从 0.18–0.30 扩展到 **0.15–0.33 m/s**；动作噪声约 0.22。最终模型在同一 16 环境评估中平均前进 **0.285 m/s**、平均绝对误差 **0.042 m/s**，仍零重置。横移、转向及更高速指令尚未完成同样的定速验证。

最终模型：[model_1497.pt](logs/rsl_rl/unitree_p_g1_rewards_pace_walk/2026-10-12_18-20-12_walk_curriculum_check_500/model_1497.pt)；20 秒回放：[rl-video-step-0.mp4](logs/rsl_rl/unitree_p_g1_rewards_pace_walk/2026-10-12_18-20-12_walk_curriculum_check_500/videos/play/rl-video-step-0.mp4)。

在当前桌面单显示器状态下，Isaac Sim 的 GUI 回放出现 Vulkan `vkCreateSwapchainKHR failed`，窗口黑屏；无界面视频回放与物理仿真、训练均正常，黑屏进程已关闭。图形驱动/显示器状态恢复后可用上面的非 `--headless` 回放命令重新打开实时界面。
