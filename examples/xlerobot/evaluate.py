from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.record import record_loop_for_xlerobot
from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 180
TASK_DESCRIPTION = "pick up bottle and open it"
MODEL_REPO_ID = "k1000dai/act_pick_and_open_bottle"
EVAL_DATASET_REPO_ID = "k1000dai/eval_pick_and_open_bottle"

# Create the robot and teleoperator configurations

robot_config = XLerobotClientConfig(remote_ip="192.168.10.203", id="my_lekiwi")
robot = XLerobotClient(robot_config)
policy = ACTPolicy.from_pretrained(MODEL_REPO_ID)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=EVAL_DATASET_REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=12,
)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()

_init_rerun(session_name="recording")

listener, events = init_keyboard_listener()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Running inference, recording eval episode {recorded_episodes} of {NUM_EPISODES}")

    # Run the policy inference loop
    record_loop_for_xlerobot(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        dataset=dataset,
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
            control_time_s=EPISODE_TIME_SEC,
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
listener.stop()
