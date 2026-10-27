# G1 deploy

A standalone C++ policy deployment runtime for the **Unitree G1 with 29 degrees of freedom**. It combines gamepad-controlled walking and running, getting up from the floor, and whole-body control driven by streamed poses.

The controller receives joint states, IMU data, and gamepad input through Unitree DDS. Policies run at **50 Hz**; commands are published through the native DDS control loop. The optional Python video pipeline estimates human poses, retargets them to G1, and streams robot pose references to the controller over UDP.

## Build the deployment controller

Requirements: a C++17 compiler, CMake, installed **Unitree SDK2**, Eigen3, yaml-cpp, nlohmann-json, Boost program_options, spdlog, and fmt. The C++ deployment runtime uses CPU inference and does not require ROS2 or CUDA.

Run from the repository root:

```bash
cmake -S deploy/g1 -B deploy/g1/build -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/g1/build -j2 -- --jobserver-style=pipe
deploy/g1/build/g1_ctrl --check-config
```

The build command above uses GNU Make. With a different CMake generator, use `cmake --build deploy/g1/build -j2`.

`--check-config` validates the FSM, policy files, and model contracts without opening DDS channels or sending robot commands.

CMake automatically selects the bundled CPU ONNX Runtime for **x86-64** or **ARM64**, including Jetson. To use another installation, configure with `-DONNXRUNTIME_ROOT=/path/to/onnxruntime`. The Python camera environment on Jetson requires its own compatible dependencies.

The controller builds from the sources in `deploy/g1/` and uses the bundled policies and deployment support. No sibling repository or Unitree RL Lab submodule is required at runtime.

## Run the deployment

Start the simulator first, then start the controller with the matching DDS interface and domain. For the current `unitree_mujoco` setup:

```bash
deploy/g1/build/g1_ctrl -n lo --domain 1
```

For a physical G1, replace `lo` and `1` with the robot network interface and DDS domain. The controller waits for `rt/lowstate` before starting the FSM. A USB/Bluetooth gamepad connected to the simulator is forwarded through DDS as a Unitree controller.

The default initial state is **Passive**. Select a mode with the gamepad:

- **L2 + Up:** enter **FixStand**, a PD transition from measured joints to the configured standing pose. This state holds joint targets; it does not run a balance policy.
- **L2 + Up again in FixStand:** start **GetUp**, then automatically enter **Walk**.
- **R1 + X:** **Walk**. The left stick controls translation; the right stick controls turning. Center the sticks to stop. No button needs to remain held.
- **R1 + Y:** **Run**, using the same TrajLoco policy with separate speed limits.
- **L1 + Up:** **Idle**, balance control with zero walking command.
- **R2 + A:** **Teleop**, tracking streamed poses for the whole observed body.
- **R2 + X:** **TeleopArms**, tracking only streamed arm and wrist poses. The policy continues to control the legs and torso for balance.
- **L2 + B:** **Passive**.

Transitions are defined in the FSM configuration. Walk commands are limited to **1.5 m/s forward**, **0.5 m/s backward**, and **1 m/s sideways**. Run limits are configured separately in `FSM.Run.velocity_limits`. These are command limits, not guaranteed physical speeds.

## Policies and configuration

The deployment includes three policy bundles:

- **TrajLoco:** walking, running, and idle balance. A velocity command is integrated into a reference path for the policy.
- **WBC:** whole-body balance and tracking of externally supplied joint pose references.
- **GetUp:** a dedicated decoder conditioned on one of two recorded get-up references, selected from the robot orientation. It uses measured robot state as feedback and transitions to Walk when finished.

Configuration files:

- [`config.yaml`](deploy/g1/config/config.yaml): FSM states, button mappings, transitions, policy paths, standing poses, gains, and per-state settings.
- [`deploy.yaml`](deploy/g1/config/deploy.yaml): shared runtime settings, including frequency, DDS state freshness, camera UDP endpoint, interpolation, and reference smoothing.
- [`robot_defaults.yaml`](deploy/g1/config/robot_defaults.yaml): robot defaults and PD torque clipping parameters.
- [`initial_pose.json`](deploy/g1/config/initial_pose.json): the initial WBC reference used by the camera sender when tracking is unavailable.
- [`policies/`](deploy/g1/policies): model weights and their original policy parameters. Copied source file checksums are recorded in `SOURCE.json`.

Use `--config /path/to/config.yaml` and `--deploy /path/to/deploy.yaml` to select another configuration.

## Receive poses from video or a live camera stream

Teleop accepts **streamed G1 pose references over UDP** using the `gmr-g1-v1` format. The included video pipeline can produce these references from a video file, a local camera, or a network video stream. Another producer can use the same protocol to supply robot joint angles, timestamps, and tracked-joint masks.

Pose estimation runs independently of deployment. Human poses are retargeted to G1 before transmission; the deployment controller receives robot references rather than raw images or raw human landmarks.

### Set up the video sender

From the repository root:

```bash
git submodule update --init vitpose
./uv sync --locked
./uv run python -m video_teleop.pose_stream --download-models
```

The video sender uses Python 3.11, Easy-ViTPose, MediaPipe, and GMR retargeting. Its environment is managed through `uv`.

### Live camera

Keep the deployment controller running and launch the sender in another terminal:

```bash
./uv run python -m video_teleop.pose_stream \
  --source 0 --retarget --udp 127.0.0.1:9870 --no-display
```

### Video file

```bash
./uv run python -m video_teleop.pose_stream \
  --source /absolute/path/video.mp4 \
  --retarget --udp 127.0.0.1:9870 --no-display
```

### Network video stream

```bash
./uv run python -m video_teleop.pose_stream \
  --source 'rtsp://camera-address/stream' \
  --retarget --udp 127.0.0.1:9870 --no-display
```

RTSP and HTTP video sources are opened through OpenCV/FFmpeg; support for a particular stream depends on the installed backend and codecs.

Select **R2 + A** for whole-body teleop or **R2 + X** for arms-only teleop. Both modes use the same sender; the deployment selects which body parts to follow.

To run the sender on another computer, set `udp_host` in `deploy.yaml` to the controller's listening interface, or `0.0.0.0`, and send with `--udp CONTROLLER_IP:9870`. The default `127.0.0.1` setup keeps both processes on one computer.

### Streaming behavior

- The sender publishes at **50 Hz**, independently of camera inference FPS. Repeated references retain the source frame identity; the controller interpolates completed poses without predicting future movement.
- **Teleop waits indefinitely for pose data.** Missing initial frames, a stream timeout, or a sender `stopped` message leave Teleop/TeleopArms active. WBC continues balance control with the initial stance. Incoming poses resume teleop smoothly without selecting the mode again.
- `stream_timeout_s` controls when an old reference is replaced by the initial stance. It does not switch the policy. Reference changes are smoothed and limited to **5 rad/s per joint**; this limit applies to reference motion, not learned balance corrections.
- Unavailable body parts are excluded from camera goals. A leg requires its complete joint chain to be observed. Arms mode keeps camera goals out of the legs and torso.
- The sender retries an unavailable camera or file and sends the initial reference while waiting. At the end of a video, it continues sending that reference without automatically replaying the file.
- Model or receiver errors still use the configured recovery to Walk. Invalid robot state, lost DDS LowState, and the Passive button retain their existing handling.

Remove `--no-display` to view one window containing the full video frame and the human skeleton. There is no robot rendering or automatic image crop. Use **Tab** to select another person and **Q/Esc** to quit. `--profile` reports processing times and FPS.

## Validation

```bash
ctest --test-dir deploy/g1/build --output-on-failure
./uv run python -m unittest discover -s deploy/tests -v
./uv run python -m unittest discover -s video_teleop/tests -v
```

The offline deployment checks use the real ONNX policies and MuJoCo without publishing commands to a physical robot.

See the [detailed deployment guide](deploy/g1/README.md) and [video teleoperation guide](video_teleop/README.md) for implementation details.

## Acknowledgements

The deployment framework is adapted from the original **[Unitree RL Lab deployment](https://github.com/unitreerobotics/unitree_rl_lab)**. Its minimal FSM, DDS, and gamepad support is included in `deploy/g1/third_party/unitree_deploy/` with the original Apache-2.0 license. Policy integration also follows the source `policy_deployment_ros2` implementation; provenance is recorded with the bundled configuration and policies.

Human pose estimation uses **[Easy-ViTPose](https://github.com/JunkyByte/easy_ViTPose)**, based on **[ViTPose](https://github.com/ViTAE-Transformer/ViTPose)**.
