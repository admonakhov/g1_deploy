# Deployment sources

Three policy bundles are deployed. They were copied unchanged from:
`policy_deployment_ros2/robots/g1_29dof_ros2_deploy/config/policy/`
in the source repository. Provenance paths are repository-relative, without local usernames or home directories.

- `vq/traj_loco` → `policies/traj_loco`, including both auxiliary networks.
  The SDK joint order, ten 75-value history frames, first-frame history padding,
  trajectory integration, 64-value anchor, 163-value command, stand-adjust
  selection, estimated odometry, and 23 residual actions follow `reset_vq`,
  `vq_step_velocity`, `vq_path`, `run_vq_policy` in the supplied
  `src/ros2_policy_deploy.cpp`.
- `wbc` → `policies/wbc`. The 132-value actor input uses the original training
  term order, scales, default-relative measured joints, previous raw action,
  and reference-relative action output. Reference semantics follow
  `WbcMotionLoader::ref_base_*`, `ref_gravity_b`, `ref_joint_pos` and the ROS2
  deploy's `compute_wbc_reference_term` / `process_action`.
- All imported policy files have SHA256 hashes in `policies/SOURCE.json`.
  These are user-provided weights; no new license claim is made for them.
- Unitree RL Lab `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` supplies the
  SDK/DDS wrappers, Passive and configurable FSM infrastructure. The standard
  G1 main and CMake delegate to this deployment. Other G1 policies are removed.
- `sim/g1_29dof.xml` and meshes were copied from
  `unitree_mujoco/unitree_robots/g1` for offline tests.
  Only the mesh location and the test scene are adapted. The BSD-3-Clause
  license is preserved in `sim/UNITREE_LICENSE`.
- ARM64 ONNX Runtime source and binary hashes: `third_party/ONNXRUNTIME_SOURCE.md`.
- GMR input and joint-limit extraction use the existing GMR G1 asset;
  its license remains under `video_teleop/third_party/gmr`.

The GR00T/SONIC implementation and dependency were removed. This port uses
native Unitree DDS, not ROS2, and adds bounded reference interpolation, observed
  body masks, measured-state entry and watchdog handling around the policies.
