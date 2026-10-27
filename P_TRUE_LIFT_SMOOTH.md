# P1 measured sole lift follow-up

## Bootstrap fine-tuning from the standing policy (2026-10-27, current)

The continuous-step run stopped near iteration 7803 with no valid steps.
`model_7800.pt` is used as a standing-policy initialization for a new 2,000
iteration trial. Loading its weights preserves the actor/critic architecture
and action scales. The optimizer, run counter and curriculum start fresh.

Starting changes:

- Planar velocity tracking initially retains its full reward. After each
  curriculum window, only measured physical gait quality above 0.8 can advance
  `bootstrap_progress`. Progress moves 25% toward the normalized gait-quality
  target and never increases merely from elapsed iterations or standing.
- As progress rises from 0 to 1, the velocity reward floor changes from 1.0
  toward 0.25, and the bootstrap reward weight fades from 0.5 toward zero.
  Existing full-gait/standing/stopping curriculum gates still apply.
- KneeLift's new `bootstrap_knee_lift` term pays incremental, capped guidance
  below the main 1 cm takeoff threshold. Opposite support must be confirmed;
  the guided foot force must be less than half the other foot's force. Knee
  flexion and knee separation contribute 40%; real sole lift contributes 60%,
  with the existing excessive hip-roll attenuation. Equal-load planted bends
  and bilateral crouches provide no knee credit.
- A phase pays at most one credit. Holding the foot, failed retries or zero-
  command switching cannot renew it. An opposite-leg phase becomes eligible
  only after a flexed 6 mm mini-lift lands with confirmed contact, at least
  60 ms unloading and a duration <= 0.8 s. Main physical gait counters still
  require >=2 cm and their original swing/support tests. Mini-lifts do not
  qualify for physical gait quality or course promotion.

Bootstrap lift target is 2.5 cm; knee progress starts at 0.01 rad and reaches
its cap at 0.18 rad with swing/support separation. Tunable settings are beside
the knee joints in `KneeLiftRewardsCfg.gait.params`. The original 6 cm full-lift
target is retained. New TensorBoard keys include
`Episode_Reward/bootstrap_knee_lift`, `Metrics/gait/left_bootstrap_steps`,
`right_bootstrap_steps`, `bootstrap_credit_paid`, and curriculum
`bootstrap_progress`, `velocity_reward_floor`, `bootstrap_reward_weight`.

`train.py --init_checkpoint PATH` is the new weight-only initialization mode;
it is mutually exclusive with `--resume`. It defaults to a fixed learning
rate <= 1e-4 and preserves the learned per-joint exploration noise. The optional
`--init_noise_std` sets a uniform std within the task bounds, but is omitted
from the recommended trial because the source's noise varies by joint.
`params/initialization.yaml` records the source, iteration and initialization
settings; new checkpoints also retain source provenance in `infos`.

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth-KneeLift --num_envs 4096 --max_iterations 2000 --seed 42 --logger tensorboard --run_name knee_lift_bootstrap_finetune_from_7800 --init_checkpoint /home/pc/PACE_P1/unitree-p-pace/logs/rsl_rl/unitree_p_g1_pace_walk_knee_lift/2026-10-26_19-12-26_knee_lift_continuous_steps_from_scratch/model_7800.pt
```

The trial's new iteration numbers are 0-1999; it writes a separate run directory.
Validate both true >=2 cm left/right steps and zero-command standing after
this trial. Positive bootstrap credit alone does not establish a walking gait.

## Continuous walking validation (2026-10-26)

The staged policy completed training with stable zero-command behavior but
mostly shuffled at 0.15 m/s. A deterministic 1,000-step evaluation measured
0.163 m/s forward speed with no resets, while only three complete lift cycles
reached 2 cm. Keyboard's former 0.55 m/s target caused nine resets. Current
training therefore couples velocity tracking to recent physical gait quality
and requires continuous walking before curriculum promotion.

`RewardsCfg.track_lin_vel_xy` keeps its original stationary objective. For a
nonzero planar command, the reward is now multiplied by
`0.25 + 0.75 * recent_quality`. The 25% floor preserves a starting signal
before the first completed steps. Quality uses a fixed 4-second ring of real
completed steps, multiplying the smaller left/right foot-rate score, the
smaller left/right mean peak-height score and physical alternation rate. Each
score is clipped to [0, 1]. Old evidence expires and standing clears the ring;
single-leg lifts cannot earn quality. No unobserved periodic phase is added.

Curriculum gating uses episode totals over actual planar-command seconds,
excluding standing and yaw-only samples. Full steps retain the existing
ground contact, sole height, air-duration and support tests. Gait scoring
uses physical cycles, not normalized reward sums or episode maximum heights:

| Stage | Minimum rate of EACH foot | Minimum mean peak of EACH foot |
| --- | --- | --- |
| 0 | 0.4 steps/s | 0.025 m |
| 1 | 0.5 steps/s | 0.040 m |
| 2 | 0.6 steps/s | 0.050 m |

All stages require >= 90% alternation and >= 100 physical transitions per
evaluation window. Windows require >= 200 accumulated planar-command seconds
across environments, in addition to the existing standing, tracking, stopping
and survival samples. `gait_score` reaches 1 only when all physical conditions
pass. Promotion requires that score; <= 0.4 contributes to demotion. Two-window
promotion/retreat rules remain in effect. The sole lift reward still targets
6 cm. Stage thresholds are in `CurriculumCfg.velocity_levels.params.stages`.

New diagnostics: `Metrics/gait/walking_time_s`, left/right
`walking_step_rate_hz`, left/right `walking_mean_step_peak_m`,
`walking_alternation_rate`, `walking_transitions`, `mean_walking_quality`.
`Curriculum/velocity_levels/` also exposes left/right `step_rate_hz`, left/right
`mean_step_peak_m`, `walking_alternation`, `evaluated_walking_seconds`, and
`evaluated_walking_transitions`. Read counts and moving time with these ratios.

Playback defaults to `--keyboard-vx 0.15 --keyboard-vy 0.05`, starts at zero
and ramps on held keys. Each new checkpoint saves its actual command ranges
and stage in `infos`; keyboard targets are capped inside these ranges. For
older checkpoints without metadata, playback reads the initial ranges from
`params/env.yaml` conservatively. A forward-only checkpoint therefore disables
S; zero lateral ranges disable A/D. The interface and terminal display the
effective limits. These playback changes also apply when loading old policies.

New experiment name: `knee_lift_continuous_steps_from_scratch`. Run from scratch
using the existing KneeLift task. Implementation checks exercise unilateral
lifts, rare steps, low heights, stale windows, standing transitions, partial
resets, stage changes, checkpoint metadata and unchanged policy export. Short
training checks establish execution and finite losses, not successful gait.

## Staged standing, walking and stopping (2026-10-25)

The current TrueLiftSmooth / KneeLift configuration adds squared position
drift (-8) and wrapped heading drift (-2) costs for standing commands. The
anchor is captured when the ramped command first enters standing and remains
fixed until moving resumes. Initial/reset anchors are masked until valid.

The same policy learns the following performance-gated stages:

| Stage | Standing probability | Moving commands | Gait/lift/air/knee weight multiplier | Push velocity x/y |
| --- | --- | --- | --- | --- |
| 0 | 50% | Forward 0.08-0.16 m/s | 0.25 | 0.05 / 0.03 m/s |
| 1 | 35% | Forward 0.08-0.24 m/s, 15% lateral modes | 0.50 | 0.15 / 0.10 m/s |
| 2 | 25% | Forward/backward, lateral, turning and combined modes | 1.00 | 0.50 / 0.50 m/s |

Stage settings and thresholds are together in `CurriculumCfg.velocity_levels`.
The 6 cm clearance target is retained. Reward scales use the task's nominal
weights, so advancing or retreating does not compound multiplication. Inactive
axes are excluded from sample and tracking gates, preventing stage 0 from
waiting for lateral/yaw samples it cannot generate. Speed ranges expand only
in the final stage. Stage changes affect future command samples; commands
already active finish normally.

Each evaluation window requires 10,000 policy steps since the last update,
20,000 samples per active axis, 10,000 standing samples, 512 completed episodes
and 50 walking-to-stop attempts. Two consecutive good windows advance one
stage. Promotion requires standing score >= 0.72, active-axis tracking scores
>= 0.75, survival >= 0.85, stable-standing fraction >= 0.80 and stopping success
>= 0.80. Two consecutive bad windows retreat one stage; regression thresholds
include standing score <= 0.45, tracking <= 0.45, survival <= 0.70, stable
fraction < 0.50 or stopping success < 0.50. Stage 0 cannot retreat further.

Stable standing requires XY speed < 0.04 m/s, yaw speed < 0.10 rad/s, anchor
position drift < 0.04 m, heading drift < 0.10 rad and projected gravity XY norm
< 0.10. A stop trial begins when a moving target becomes standing; it succeeds
after 0.5 continuous stable seconds within 2 seconds of the target change.
Initial standing after reset is not counted as a stop trial. Attempts cut
short by a fall or another moving command remain unsuccessful.

TensorBoard diagnostics:

- `Metrics/base_velocity/standing_xy_speed_m_s`, `standing_yaw_speed_rad_s`,
  `standing_position_drift_m`, `standing_heading_drift_rad`: sample-weighted
  within the batch of resetting episodes, excluding moving samples.
- `standing_samples`, `standing_stable_fraction`, `standing_max_stable_time_s`,
  `stopping_attempts`, `stopping_success_fraction`, `stopping_settle_time_s`
  under the same metric prefix. Zero samples/attempts produce zero metrics;
  read the denominators before interpreting the values. Settle time averages
  successful trials only and includes the stable dwell time.
- `Curriculum/velocity_levels/stage`, `standing_fraction`, `lift_reward_scale`,
  `good_windows`, `bad_windows`, `standing_stable_fraction`, `stopping_success`.
- `Episode_Reward/standing_position_drift`, `standing_heading_drift`.

Checks: `python scripts/check_p_stability.py --headless` and
`python scripts/check_p_alternation.py --headless`. Run a fresh experiment:

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth-KneeLift --num_envs 4096 --max_iterations 30000 --seed 42 --logger tensorboard --run_name knee_lift_staged_stand_walk_from_scratch
```

Short checks validate implementation; successful standing/walking still needs
to be established from the resulting policy. The sections below describe
earlier experiments and nominal reward weights.

## Standing command update (2026-10-23)

TrueLiftSmooth / KneeLift now sample standing commands with probability 0.25.
Velocity curriculum promotion requires at least 10,000 standing samples and a
standing score >= 0.72; score <= 0.45 permits demotion. The existing score
includes base XY speed, yaw speed, position drift and heading drift.

Additional standing-only squared costs: base XY speed (-4), base yaw speed
(-1), joint speed sum (-0.02), action change sum (-0.10), and contacting-foot
world yaw speed sum (-0.20). These are continuous costs integrated with dt.
They require both a standing target and command norm below 0.04, excluding
moving-command acceleration ramps and intentional yaw commands. Foot contact
uses the sensor force history with a 1 N threshold. Rewards appear separately
as `Episode_Reward/standing_*` in TensorBoard.

Start a new run with the KneeLift task and run name
`knee_lift_stand25_stability_from_scratch`. Validate both zero-command standing
and nonzero-command walking; these changes do not establish convergence or
resolve the previously measured asymmetric knee flexion by themselves.

P1 的全部环境配置统一在
[`p_pace/velocity_env_cfg.py`](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/p_pace/velocity_env_cfg.py)。
当前任务直接使用 `RobotEnvCfg` / `RobotPlayEnvCfg`，继承 Isaac Lab 的基础环境。
按以下位置调参即可，无需跨环境文件查找：

| 配置类 | 内容 |
| --- | --- |
| `RobotSceneCfg` | P1 模型、脚踝 PD 增益、平地、接触传感器、灯光 |
| `EventCfg` | 材质/质量随机化、重置、外力与推扰动 |
| `CommandsCfg` | 指令采样、速度范围、加减速度 |
| `ActionsCfg` | 动作缩放、关节目标限幅、控制延迟 |
| `ObservationsCfg` | Actor/Critic 的观测顺序、噪声、15 帧历史 |
| `RewardsCfg` | 全部奖励函数、权重和参数 |
| `TerminationsCfg` | 超时、跌倒和碰撞终止 |
| `CurriculumCfg` | 速度课程 |
| `RobotEnvCfg` / `RobotPlayEnvCfg` | 训练/播放组合、仿真步长和环境数量 |

历史任务在该文件末尾定义，并继续通过原任务 ID 加载。原来分散的
`reference_cfg.py`、`walking_env_cfg.py`、`g1_walk_env_cfg.py`、
`no_phase_15_env_cfg.py`、`dense_lift_env_cfg.py`、`true_lift_smooth_env_cfg.py`
已合并。调节当前奖励请修改前半部分的 `RewardsCfg`。

当前左右脚踝的 pitch、roll 四个关节使用 `Kp=40`、`Kd=2`，在
`RobotSceneCfg.robot.actuators` 中设置；髋、膝关节使用 `Kp=180`、`Kd=10`。
新启动的训练、播放和后续部署导出会读取这些增益。此前训练日志中的
`deploy.yaml` 仍记录生成时的增益，评估旧策略时需注意增益已改变。

## KneeLift：屈膝抬脚与摆动腿侧摆约束

新任务为 `Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth-KneeLift`。
仍在同一份 `velocity_env_cfg.py` 中调整：`KneeLiftActionsCfg`、
`KneeLiftRewardsCfg`、`KneeLiftEnvCfg` / `KneeLiftPlayEnvCfg`。
原 TrueLiftSmooth 及其他历史任务的配置保持原值，旧权重继续使用原任务播放。

修改依据：原最终策略以 0.25 m/s 播放时，在每步离地最高点，左右膝相对默认姿态
增量中位数为 0.3072 / 0.3443 rad，髋侧摆增量为 +0.1622 / -0.1571 rad。
左右膝已接近旧 0.36 rad 动作增量上限，不能直接判断为完全不屈膝。
关节角统计可描述动作配合，不能单独证明各关节对离地高度的因果贡献。

| 参数 | KneeLift 设置 |
| --- | --- |
| 膝动作缩放 / 偏移限幅 | 0.55 / [-0.22, 0.55] rad |
| 离地高度 / 有效迈步奖励 | 权重 +2.0 / +1.5；仍使用原 6 cm 高度目标和每轮交替额度 |
| 屈膝奖励 | 权重 +1.0；膝相对默认姿态增量从 0.08 到 0.42 rad 递增后封顶 |
| 摆动/支撑膝差 | 两腿相对各自默认姿态的屈曲增量差需从 0.04 到 0.26 rad 递增，防止双膝持续蹲伏得分 |
| 侧摆配合 | 屈膝奖励乘以真实离地进度和 `exp(-侧摆超限平方)` |
| 摆动腿侧摆惩罚 | 权重 -0.5；直行容许 0.08 rad，超限以 0.10 rad 归一化平方 |
| 侧移 / 转向放宽量 | 最大额外 0.08 / 0.04 rad；按实际指令连续变化 |
| 落脚屈膝惩罚 | 权重 -0.15；真实有效落脚时只惩罚超过默认姿态 +0.16 rad 的部分 |

新正奖励与原交替状态共享，每腿每轮最多一份额度；保持悬腿、接触抖动和同腿重试
不会重新发放。侧摆惩罚只作用于摆动腿，落脚约束不要求膝盖完全伸直。
保留 6 cm 脚底目标、交替规则、脚踝 Kp40/Kd2 和 0.05–0.35 探索噪声限制。

新 TensorBoard 指标位于 `Metrics/gait/`：左右 `peak_knee_delta_rad`、
`peak_knee_separation_rad`、`peak_hip_roll_abs_rad`、`peak_body_roll_abs_rad` 和 `knee_flexed_step_fraction`。
这些指标取完整有效迈步中、真实脚底最高点的关节姿态；无有效步时记 0，需结合步数阅读。
`scripts/eval_p_gait.py` 也会打印每步最高点的膝角、髋侧摆和机身侧倾中位数。

从头训练：

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth-KneeLift --num_envs 4096 --max_iterations 30000 --seed 42 --logger tensorboard --run_name knee_lift_strong_clearance_from_scratch
```

新日志写入 `logs/rsl_rl/unitree_p_g1_pace_walk_knee_lift/`。
新权重的动作缩放与旧策略不同，播放和导出必须选择新 `KneeLift` 任务。

2026-10-20 的 `11-08-02_knee_lift_low_roll_from_scratch` 在约 1.3 万至
1.85 万轮之间仍几乎没有有效交替迈步：左/右回合离地峰值约 2.7/1.4 cm，
右脚每回合有效步数约 0.003。虽然总奖励从 38.04 小幅增长到 39.47，
步态指标没有实质改善。因此下一轮只调整三项正奖励权重：
离地 0.5→2.0、迈步 0.5→1.5、屈膝 0.35→1.0。
保留侧摆和落脚约束、动作范围、速度跟踪权重与探索噪声范围。
新的总奖励数值受权重影响，不可直接与旧轮总奖励比较；应比较有效步数、交替率和实测每步高度。

The completed DenseLift checkpoint `model_29999.pt` was evaluated in Isaac at
`vx=0.25 m/s` for 1000 steps, with 100 warmup steps and no resets. Both feet had
3 mm median peaks in contact-free episodes, far below the 60 mm target. The
contact sensor reported 118 left and 31 right contact-free episodes of at least
80 ms during 18 s of commanded walking. These counts include shallow foot
unloading and are not counts of full walking strides. The measured action-vector
step change was 0.947 RMS at the 50 Hz policy rate, and body roll/pitch angular
speed was 0.201 rad/s RMS.

The DenseLift reward used a 4 cm Gaussian width, so a 2 cm foot lift at the
6 cm swing target still earned about 37% of the maximum height match. The
reward also accepted any sample with zero `current_contact_time` as airborne.
Brief contact loss could therefore earn some height reward with very little
physical lift.

The current `NoPhase15-TrueLiftSmooth` task multiplies planar velocity tracking
by actual directed motion progress. At a nonzero movement command, zero measured
velocity earns zero planar tracking reward. At a zero planar command, the usual
stationary tracking objective is retained.

### Alternating physical footsteps

The motion-progress run's `model_7400.pt` tracked a 0.248 m/s command at
0.247 m/s, but completed 25 left steps and zero right steps. Its learned action
noise had also grown to a mean standard deviation of approximately 50.
Reward growth therefore did not establish a usable alternating gait.

`RewardsCfg.gait` now owns one shared left/right step state. A swing starts
after confirmed ground support and at least 1 cm sole clearance. A completed
step must reach at least 2 cm, spend 120-600 ms airborne, and land within 5 mm
of the plane for 60 ms with a ground contact observation. Foot height is the
lowest collision-sphere surface, accounting for foot rotation. Both feet
airborne invalidate the swing. Rewarded landings must be at least 250 ms apart.

Either foot can start. After a rewarded landing, only the opposite foot can
earn the next gait, lift or air-time reward. Same-leg physical steps are still
counted in diagnostics so the alternation metric cannot hide them. Partial
episode resets clear only the affected environments' state.

Lift credit is paid only for new height progress toward 6 cm, with a fixed
2 mm floor. Holding a foot up or retrying the same foot cannot repeatedly
collect this credit. Its budget renews only after the opposite foot completes
an accepted step. Gait credit and air-time credit are paid at the accepted
landing; air-time credit is capped at 400 ms. These event credits are divided
by policy `dt` before RewardManager applies its usual `weight * dt`.
Gait, lift and air-time weights are each 0.5. Action-rate weight remains -0.03,
body roll/pitch angular-speed weight remains -0.10, and ankle gains remain
Kp40/Kd2. The air-time variance penalty is retained.

The TrueLiftSmooth PPO configuration sets `min_noise_std=0.05`,
`max_noise_std=0.35` and `entropy_coef=0.001`. The runner clamps noise after
each optimizer step and checkpoint load. The actor/critic network, observation
order and exported deterministic policy format stay the same. The bounds
control exploration noise; the action mean can still saturate and is monitored
through the rollout action clipping fraction.

TensorBoard records episode means under:

- `Metrics/gait/left_valid_steps`, `right_valid_steps`: all physical valid cycles.
- `Metrics/gait/left_rewarded_steps`, `right_rewarded_steps`: accepted cycles.
- `Metrics/gait/alternation_rate`: alternating transitions / all valid transitions.
- `Metrics/gait/step_transitions`, `same_leg_repeats`: denominators and violations.
- `Metrics/gait/left_clearance_peak_m`, `right_clearance_peak_m`: measured sole peaks.
- `Policy/min_noise_std`, `Policy/max_noise_std`, `Policy/action_clip_fraction`.

Zero transitions produce a zero alternation rate; always read the counts with
the ratio. Run `python scripts/check_p_alternation.py --headless` for checks of
step ordering, prolonged holds, contact flicker, failed retries, jumps, partial
resets, noise bounds and deterministic JIT export. Convergence still needs to
be assessed in a new training run.

Train from scratch in a separate experiment directory:

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth --num_envs 4096 --max_iterations 30000 --seed 42 --logger tensorboard --run_name alternating_steps_bounded_noise_from_scratch
```

Measure a resulting checkpoint at a fixed forward command:

```bash
python scripts/eval_p_gait.py --headless --task Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth --checkpoint logs/rsl_rl/unitree_p_g1_pace_walk_true_lift_smooth/RUN_NAME/model_29999.pt --steps 1000 --warmup-steps 100 --vx 0.25
```

The evaluator reports actual left/right lift-land cycles, the physical step
sequence and alternation rate, clearance statistics, body angular speed,
action changes, speed tracking and resets. Use these measurements to judge
the new policy. The earlier 30,000-iteration TrueLiftSmooth run completed with checkpoint
`logs/rsl_rl/unitree_p_g1_pace_walk_true_lift_smooth/2026-10-17_18-51-36_true_lift_smooth_from_scratch/model_29999.pt`.
Its final 1,000-iteration mean clearance reward was 0.0063; physical clearance
still needs to be measured. Configuration consolidation was checked against all
six task variants in both training and playback, including observation order.
