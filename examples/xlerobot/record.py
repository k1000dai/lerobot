from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop_for_xlerobot
from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
# from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

NUM_EPISODES = 3
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10
REPO_ID = "k1000dai/test"
TASK_DESCRIPTION = "My task description"

robot_config = XLerobotClientConfig(remote_ip="192.168.10.203", id="my_lekiwi")
# teleop_arm_config = SO100LeaderConfig(port="/dev/tty.usbmodem585A0077581", id="my_awesome_leader_arm")
teleop_left_arm_config = SO101LeaderConfig(port="/dev/tty.usbmodem5A7A0176981", id="my_awesome_left_leader_arm")
teleop_right_arm_config = SO101LeaderConfig(port="/dev/tty.usbmodem5A7A0178511", id="my_awesome_right_leader_arm")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = XLerobotClient(robot_config)
leader_left_arm = SO101Leader(teleop_left_arm_config)
leader_right_arm = SO101Leader(teleop_right_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
leader_left_arm.connect()
leader_right_arm.connect()
keyboard.connect()

_init_rerun(session_name="lekiwi_record")

listener, events = init_keyboard_listener()


if not robot.is_connected or not leader_left_arm.is_connected or not leader_right_arm.is_connected or not keyboard.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {recorded_episodes}")

    # Run the record loop
    record_loop_for_xlerobot(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        left_arm=leader_left_arm,
        right_arm=leader_right_arm,
        keyboard=keyboard,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Logic for reset env
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("Reset the environment")
        record_loop_for_xlerobot(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            left_arm=leader_left_arm,
            right_arm=leader_right_arm,
            keyboard=keyboard,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    recorded_episodes += 1

# Upload to hub and clean up
dataset.push_to_hub()

robot.disconnect()
leader_left_arm.disconnect()
leader_right_arm.disconnect()
keyboard.disconnect()
listener.stop()
