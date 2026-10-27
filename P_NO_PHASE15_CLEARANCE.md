# P1 NoPhase15: 6 cm sole clearance and slower reference gait

Task: `Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15`.
The original phased tasks and archived checkpoints are unchanged.

- Actor: 675 inputs (45 features x 15 frames), 12 actions; critic: 720 inputs.
- PACE CSV, encoder convention, motor curves, PD gains, URDF and domain randomization are unchanged.
- Gait/contact reward and swing-clearance reward use a shared 1.0 s reference cycle.
- Swing reference: `0.06 * sin(pi * swing_progress)^2` meters above flat ground.
- Height error kernel: `exp(-((clearance - reference) / 0.02)^2)`.
- Clearance uses the lowest of the four foot collision spheres, including foot rotation and sphere radius.
  Geometry matches `urdf_803_nobattery/urdf/p1.urdf`; this reward assumes a plane at each environment origin's z.
- Clearance reward retains weight 1.0, applies during commanded airborne swing and emphasizes mid-swing.
- Single-support duration uses Isaac Lab's `feet_air_time_positive_biped`, weight 0.5 and cap 0.4 s.
  This cap saturates the bonus; it is not a hard duration constraint. The function gates on planar movement commands.
- Left/right completed air/contact duration variance retains weight -1.0.

Start a separate full run (this does not update an already-running training process or an existing policy):

```bash
conda activate isaaclab
cd /home/pc/PACE_P1/unitree-p-pace
python scripts/rsl_rl/train.py --headless \
  --task Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15 \
  --num_envs 4096 --max_iterations 30000 --seed 42 \
  --run_name sole6cm_period1_from_scratch
```

Validation: 8 environments / 3 PPO iterations completed in `2026-10-15_22-22-47_sole6cm_period1_smoke`.
Checkpoint and exported deploy configuration retain 675/720 input dimensions and the PACE CSV hash.
Numerical checks cover 6 cm mid-swing, insufficient/excessive clearance, tilted feet, ground offsets,
zero commands, contact gating and left/right symmetry. Original task configuration remains 5-frame / 0.8 s.

These checks validate implementation, not convergence. After full training, compare actual sole-clearance
peaks, left/right contact and air durations, touchdown frequency, velocity tracking and falls at fixed speeds.
The 1.0 s reference cycle guides the rewards; a phase-free actor is not guaranteed to follow it exactly.

## Keyboard playback

Graphical `play.py` for a P task starts with zero velocity command. Click the Isaac Sim viewport,
then hold W/S to move forward/backward or A/D to move left/right in the robot's body frame.
The command ramps up to the configured limits and ramps back to zero when the keys are released;
opposite keys cancel. The current limits are +0.55/-0.45 m/s forward/backward and
+0.30/-0.30 m/s left/right. These are command limits, not guaranteed achieved speeds.
The `P1 Keyboard Control` panel shows held keys, the requested target, the
ramped command sent to the policy, and measured body velocity. Terminal output
also reports key events and velocity values. Held-key state is polled every
simulation step. If `Keys` stays `none`, focus the Isaac window and switch to
an English keyboard layout. If `Command` increases but `Measured` stays near
zero, the policy is receiving a movement command but is not following it.

For the latest no-phase checkpoint:

```bash
conda activate isaaclab
cd /home/pc/PACE_P1/unitree-p-pace
python scripts/rsl_rl/play.py \
  --task Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15 \
  --num_envs 1 --real-time \
  --checkpoint logs/rsl_rl/unitree_p_g1_pace_walk_no_phase_15/2026-10-16_13-31-40_sole6cm_period1_from_scratch/model_11700.pt
```

Headless playback keeps the task's original fixed playback command and does not use keyboard input.

## Dense lift follow-up

The original run already includes `feet_air_time_positive_biped` at weight 0.5. That term
rewards single-support air/contact duration up to 0.4 s; it does not measure sole height.
The original 2 cm Gaussian height kernel remained near zero during training when the foot
barely lifted. The separate `NoPhase15-DenseLift` task keeps the air-time term and 6 cm
height target, widens the height kernel to 4 cm, and raises its weight from 1.0 to 2.0.
At the middle of the desired swing, a 2 cm airborne sole changes from about 0.018 to
about 0.736 weighted reward. This is an experimental reward change; it cannot alter an
already trained checkpoint without further training. Its logs use a separate experiment folder.

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-DenseLift \
  --num_envs 4096 --max_iterations 30000 --seed 42 \
  --run_name dense_lift_from_scratch
```

Compare actual left/right sole-height peaks, contact times, speed tracking and falls against
the original checkpoint before considering the new policy for deployment.
