#pragma once
#include "G1Constants.hpp"

namespace teleop {
// Quintic smoothstep: zero blend velocity and acceleration at both endpoints.
// A moving target is allowed; the separate motor speed limiter bounds the
// resulting command, including changes of target during the transition.
class JointTransition {
    Joints start_ = standing();
    double began_ = 0., duration_;
public:
    explicit JointTransition(double duration = 2.) : duration_(duration) {
        if (!std::isfinite(duration) || duration <= 0.) throw std::invalid_argument("Transition duration must be positive");
    }
    void begin(const Joints& measured, double now) {
        if (!measured.allFinite() || !std::isfinite(now)) throw std::invalid_argument("Invalid measured joint state");
        start_ = measured; began_ = now;
    }
    double blend(double now) const {
        const double u = std::clamp((now - began_) / duration_, 0., 1.);
        return u * u * u * (10. + u * (-15. + 6. * u));
    }
    double blend_velocity(double now) const {
        const double u = std::clamp((now - began_) / duration_, 0., 1.);
        return 30. * u * u * (1. - u) * (1. - u) / duration_;
    }
    Joints sample(const Joints& target, double now) const {
        return start_ + blend(now) * (target - start_);
    }
};
} // namespace teleop
