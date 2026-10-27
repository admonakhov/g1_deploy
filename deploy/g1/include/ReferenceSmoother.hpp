#pragma once
#include <algorithm>
#include "G1Constants.hpp"

namespace teleop {
// Reference-side EMA, as in Live Camera Teleop's SomaToSmpl. Time-based
// weighting keeps smoothing consistent when the camera FPS changes.
class ReferenceSmoother {
    double tau_, speed_;
    bool initialized_ = false;
    std::array<Reference, 10> previous_;
public:
    void reset(){initialized_=false;}
    explicit ReferenceSmoother(double time_constant = .25, double speed = 3.)
        : tau_(time_constant), speed_(speed) {
        if (!std::isfinite(tau_) || tau_ < 0. || !std::isfinite(speed_) || speed_ <= 0.)
            throw std::invalid_argument("Invalid reference smoothing parameters");
    }
    std::array<Reference, 10> sample(std::array<Reference, 10> frames, double dt) {
        if (!std::isfinite(dt) || dt < 0.) throw std::invalid_argument("Invalid reference time step");
        if (!initialized_) { previous_ = frames; initialized_ = true; return frames; }
        const double weight = tau_ > 0. ? -std::expm1(-dt / tau_) : 1.;
        const double delta = speed_ * std::min(dt, .05);
        for (int frame = 0; frame < 10; ++frame) {
            auto& next = frames[frame];
            const auto& before = previous_[frame];
            for (int joint = 0; joint < 29; ++joint) {
                if (!next.active[joint]) {
                    // Freeze all future slots at the current reference,
                    // rather than retaining a pre-occlusion future motion.
                    next.q[joint] = previous_[0].q[joint]; next.dq[joint] = 0.;
                } else if (dt == 0.) {
                    next.q[joint] = before.q[joint]; next.dq[joint] = 0.;
                } else {
                    const double step = std::clamp(weight * (next.q[joint] - before.q[joint]), -delta, delta);
                    next.q[joint] = before.q[joint] + step;
                    next.dq[joint] = step / dt;
                }
            }
            next.root = before.root.slerp(weight, next.root).normalized();
        }
        previous_ = frames;
        return frames;
    }
};
} // namespace teleop
