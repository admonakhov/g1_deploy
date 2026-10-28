#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "G1Constants.hpp"

namespace teleop {
constexpr double control_tilt_limit_rad = 1.;

inline double tilt_angle(const Eigen::Quaterniond& root) {
    if (!root.coeffs().allFinite() || root.norm() < .5 || root.norm() > 1.5)
        throw std::runtime_error("Invalid G1 IMU quaternion (expected WXYZ)");
    const Eigen::Vector3d up = root.normalized() * Eigen::Vector3d::UnitZ();
    return std::acos(std::clamp(up.z(), -1., 1.));
}

inline void validate_robot_state(const RobotState& robot, bool check_orientation = true,double tilt_limit=control_tilt_limit_rad) {
    const double tilt = tilt_angle(robot.root);
    if (!robot.q.allFinite() || !robot.dq.allFinite() || !robot.angular_velocity.allFinite())
        throw std::runtime_error("Invalid G1 motor/IMU state");
    if (!std::isfinite(tilt_limit)||tilt_limit<=0.||tilt_limit>M_PI)throw std::runtime_error("Invalid bad_orientation_limit_rad");
    if (check_orientation && tilt > tilt_limit) {
        std::ostringstream message;
        message << std::fixed << std::setprecision(1) << "G1 tilt " << tilt * 180. / M_PI
                << " deg exceeds " << tilt_limit * 180. / M_PI
                << " deg control limit; restore upright posture before enabling Policy/Camera";
        throw std::runtime_error(message.str());
    }
}

} // namespace teleop
