import time

from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
# from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

# Create the robot and teleoperator configurations
robot_config = XLerobotClientConfig(remote_ip="192.168.10.203", id="my_lekiwi")
# teleop_arm_config = SO100LeaderConfig(port="/dev/tty.usbmodem585A0077581", id="my_awesome_leader_arm")
teleop_left_arm_config = SO101LeaderConfig(port="/dev/tty.usbmodem5A7A0176981", id="my_awesome_left_leader_arm")
teleop_right_arm_config = SO101LeaderConfig(port="/dev/tty.usbmodem5A7A0178511", id="my_awesome_right_leader_arm")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = XLerobotClient(robot_config)
leader_left_arm = SO101Leader(teleop_left_arm_config)
leader_right_arm = SO101Leader(teleop_right_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
leader_left_arm.connect()
leader_right_arm.connect()
keyboard.connect()

_init_rerun(session_name="lekiwi_teleop")

if not robot.is_connected or not leader_left_arm.is_connected or not leader_right_arm.is_connected or not keyboard.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()
    left_arm_action = leader_left_arm.get_action()
    left_arm_action = {f"left_arm_{k}": v for k, v in left_arm_action.items()}
    right_arm_action = leader_right_arm.get_action()
    right_arm_action = {f"right_arm_{k}": v for k, v in right_arm_action.items()}
    
    arm_action = {**left_arm_action, **right_arm_action}
    
    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    log_rerun_data(observation, {**arm_action, **base_action})

    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

    robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
