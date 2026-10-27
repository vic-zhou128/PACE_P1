"""Load the P1 identification export without importing the simulator."""

import csv
import math
import os
from pathlib import Path


PARAMETER_FILE = Path(os.environ.get(
    "P_PACE_PARAMETERS_CSV",
    str(Path(__file__).resolve().parents[6] / "p1_parameters_old_vs_new.csv"),
)).expanduser().resolve()
JOINT_NAME_MAP = {
    f"{side}_{joint}": f"{side}_{joint}_{'joint' if (side, joint) == ('left', 'knee_pitch') else 'link'}"
    for side in ("left", "right")
    for joint in ("hip_roll", "hip_pitch", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
}
PARAMETER_NAME_MAP = {
    "armature": "armature",
    "黏性摩擦": "viscous_friction",
    "静/动摩擦系数": "friction",
    "bias": "encoder_bias",
}
PARAMETER_UNITS = {
    "armature": "kg m^2", "黏性摩擦": "N m s/rad", "静/动摩擦系数": "系数", "bias": "rad",
}


def load_pace_parameters(path: str | Path = PARAMETER_FILE) -> tuple[dict[str, dict[str, float]], float]:
    """Return exact-name parameter dictionaries and the unquantized fitted delay.

    The local PACE optimizer uses ``.to(torch.int)`` for delay, i.e. truncation.
    Names, units, duplicate rows and completeness are checked before training.
    """
    parameters = {name: {} for name in PARAMETER_NAME_MAP.values()}
    fitted_delay = None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {"关节", "参数", "单位", "新版"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {required_columns - set(reader.fieldnames or [])}")
        for row in reader:
            joint, name = row["关节"], row["参数"]
            value = float(row["新版"])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite identification value: {joint}/{name}")
            if name == "P1_FITTED_DELAY":
                if joint != "全部关节" or row["单位"] != "仿真步" or fitted_delay is not None or value < 0:
                    raise ValueError("Invalid or duplicate fitted delay")
                fitted_delay = value
                continue
            if joint not in JOINT_NAME_MAP or name not in PARAMETER_NAME_MAP:
                raise ValueError(f"Unknown identification parameter: {joint}/{name}")
            if row["单位"] != PARAMETER_UNITS[name]:
                raise ValueError(f"Unexpected unit for {joint}/{name}: {row['单位']}")
            target = parameters[PARAMETER_NAME_MAP[name]]
            sim_joint = JOINT_NAME_MAP[joint]
            if sim_joint in target:
                raise ValueError(f"Duplicate identification parameter: {joint}/{name}")
            if name != "bias" and value < 0:
                raise ValueError(f"Negative physical parameter: {joint}/{name}")
            target[sim_joint] = value
    for name, values in parameters.items():
        if set(values) != set(JOINT_NAME_MAP.values()):
            raise ValueError(f"Incomplete {name}: missing {set(JOINT_NAME_MAP.values()) - set(values)}")
    if fitted_delay is None:
        raise ValueError("Missing P1_FITTED_DELAY")
    return parameters, fitted_delay


P1_PARAMETERS, P1_FITTED_DELAY = load_pace_parameters()
P1_IDENTIFICATION_DT_S = 0.001  # 1000 Hz, confirmed by the user.
P1_FITTED_DELAY_STEPS = int(P1_FITTED_DELAY)
P1_DELAY_STEPS = 0  # User explicitly requests zero actuator delay for training.

# Copied from the P1_PACE_ACTUATORS in the re-extracted identification config.
# saturation_effort: stall/peak torque [N m]; effort_limit: output cap [N m];
# velocity_limit: motor no-load speed [rad/s], not a PhysX joint-speed hard cap.
P1_MOTOR_LIMITS = {
    "hip_roll": {"saturation_effort": 281.6, "effort_limit": 120.0, "velocity_limit": 16.57},
    "hip_pitch": {"saturation_effort": 281.6, "effort_limit": 120.0, "velocity_limit": 16.57},
    "hip_yaw": {"saturation_effort": 110.25, "effort_limit": 60.0, "velocity_limit": 20.56},
    "knee_pitch": {"saturation_effort": 110.25, "effort_limit": 60.0, "velocity_limit": 20.56},
    "ankle_pitch": {"saturation_effort": 181.0, "effort_limit": 68.0, "velocity_limit": 8.75},
    "ankle_roll": {"saturation_effort": 181.0, "effort_limit": 68.0, "velocity_limit": 8.75},
}
P1_MOTOR_LIMITS_BY_JOINT = {
    sim_name: P1_MOTOR_LIMITS[fit_name.split("_", 1)[1]].copy()
    for fit_name, sim_name in JOINT_NAME_MAP.items()
}
P1_MOTOR_CONFIG_SOURCE = (
    "pace-sim2real/source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/p1_pace_env_cfg.py"
)
