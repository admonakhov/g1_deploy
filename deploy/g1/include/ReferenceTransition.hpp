#pragma once
#include "JointTransition.hpp"

namespace teleop {
// Blend the motion request, not the learned balance corrections. Damping or
// rate-limiting the learned policy actions changes its trained closed-loop dynamics.
class ReferenceTransition {
    JointTransition joints_;
    Joints start_ = standing();
    Eigen::Quaterniond root_ = Eigen::Quaterniond::Identity();
public:
    explicit ReferenceTransition(double duration = 2.) : joints_(duration) {}
    void begin(const RobotState& measured, const Reference& target, double now) {
        start_ = measured.q; joints_.begin(start_, now);
        // Express measured orientation in the reference's heading frame so
        // the reference initial heading alignment is preserved through the blend.
        const auto alignment = heading(measured.root) * heading(target.root).conjugate();
        root_ = (alignment.conjugate() * measured.root).normalized();
    }
    std::array<Reference, 10> sample(std::array<Reference, 10> frames, double now) const {
        for (int i = 0; i < 10; ++i) {
            auto& frame = frames[i];
            const double time = now + i * .1;
            const double weight = joints_.blend(time);
            frame.dq = weight * frame.dq + joints_.blend_velocity(time) * (frame.q - start_);
            frame.q = joints_.sample(frame.q, time);
            frame.root = root_.slerp(weight, frame.root).normalized();
        }
        return frames;
    }
};
} // namespace teleop
