#pragma once

#include <array>
#include <string>
#include <vector>
#include <Eigen/Geometry>

namespace teleop {
using Joints = Eigen::Matrix<double, 29, 1>;
using Mask = std::array<bool, 29>;
inline const std::array<std::string, 29> motor_names = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
};
inline Joints standing() {
    // Camera fallback and WBC defaults, in Unitree SDK motor order.
    Joints q; q << -.1,0,0,.3,-.2,0, -.1,0,0,.3,-.2,0, 0,0,0,
        .35,.18,0,.87,0,0,0, .35,-.18,0,.87,0,0,0;
    return q;
}
inline Eigen::Quaterniond heading(const Eigen::Quaterniond& q) {
    const auto forward = q * Eigen::Vector3d::UnitX();
    return Eigen::Quaterniond(Eigen::AngleAxisd(std::atan2(forward.y(), forward.x()), Eigen::Vector3d::UnitZ()));
}
inline std::array<float, 6> rotation6d(const Eigen::Quaterniond& q) {
    const auto r = q.normalized().toRotationMatrix();
    return {float(r(0,0)), float(r(0,1)), float(r(1,0)), float(r(1,1)), float(r(2,0)), float(r(2,1))};
}
struct Reference {
    Joints q = standing(), dq = Joints::Zero();
    Mask active{};
    Eigen::Quaterniond root = Eigen::Quaterniond::Identity();
    Eigen::Vector3d position{0.,0.,.7842};
    bool base_tracked=false;
    double source_time = 0., alpha = 0., delay = 0.;
    int64_t previous_frame = -1, next_frame = -1;
    bool ready = false, stale = false, stopped = false;
};
struct RobotState {
    Joints q = standing(), dq = Joints::Zero();
    Eigen::Quaterniond root = Eigen::Quaterniond::Identity();
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};
} // namespace teleop
