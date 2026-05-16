#!/usr/bin/env bash
# Run MolmoAct2-SO100_101 on a real SO-101 follower arm.

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

TASK="pick up bule candy and place in bowl"
DURATION="240"

# Camera config (from previous SO-101 setup):
#   wrist: USB HD camera (/dev/v4l/by-id/usb-HD_USB_Camera_...)
#   top  : Intel RealSense D435i serial 135122073127
CAMERAS='{ wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera_2020042508-video-index0, width: 640, height: 480, fps: 30}, top: {type: intelrealsense, serial_number_or_name: "135122073127", width: 640, height: 480, fps: 30}}'
#CAMERAS='{ wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera_2020042508-video-index0, width: 640, height: 480, fps: 30}}'
#CAMERAS='{ wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera_2020042508-video-index0, width: 640, height: 480, fps: 30},top: {type: intelrealsense, serial_number_or_name: "728612070893", width: 640, height: 480, fps: 30}}'
exec lerobot-rollout \
  --strategy.type=base \
  --policy.type=molmoact2 \
  --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
  --policy.norm_tag=so100_so101_molmoact2 \
  --policy.apply_so101_calibration_compat=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.enable_cuda_graph=true \
  --policy.normalize_language=true \
  --policy.action_mode="continuous" \
  --policy.num_steps=10 \
  --robot.type=so101_follower \
  --robot.port=/dev/follower_arm \
  --robot.id=molmo_follower_arm \
  --robot.cameras="${CAMERAS}" \
  --task="${TASK}" \
  --duration="${DURATION}" \
  --fps=30 
