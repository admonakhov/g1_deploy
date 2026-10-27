#pragma once
#include <fstream>
#include <nlohmann/json.hpp>
#include "G1Constants.hpp"

namespace teleop {
class CommandLimits {
    Joints minimum_, maximum_;
public:
    explicit CommandLimits(const std::string& file) {
        std::ifstream input(file);
        nlohmann::json limits; input >> limits;
        for (int i = 0; i < 29; ++i) {
            minimum_[i] = limits.at(motor_names[i]).at(0).get<double>();
            maximum_[i] = limits.at(motor_names[i]).at(1).get<double>();
            if (!std::isfinite(minimum_[i]) || !std::isfinite(maximum_[i]) || minimum_[i] >= maximum_[i])
                throw std::runtime_error("Invalid G1 joint limits");
        }
    }
    Joints clamp(const Joints& target) const {
        if (!target.allFinite()) throw std::runtime_error("Invalid motor command");
        return target.cwiseMax(minimum_).cwiseMin(maximum_);
    }
    Joints apply(const Joints& target, const Joints& previous, const Mask& active, double speed, double dt) const {
        if (!target.allFinite() || !previous.allFinite() || !std::isfinite(dt) || !std::isfinite(speed) || speed <= 0.)
            throw std::runtime_error("Invalid motor command");
        Joints result = previous;
        const double delta = speed * std::clamp(dt, 0., .2);
        for (int i = 0; i < 29; ++i) if (active[i]) {
            // First constrain target to motor limits, then bound the movement.
            // This also avoids a jump if the measured initial pose is outside a limit.
            const double bounded = std::clamp(target[i], minimum_[i], maximum_[i]);
            result[i] = std::clamp(bounded, previous[i] - delta, previous[i] + delta);
        }
        return result;
    }
};
} // namespace teleop
