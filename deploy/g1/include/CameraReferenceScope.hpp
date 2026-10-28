#pragma once
#include <stdexcept>
#include "G1Constants.hpp"

namespace teleop {
// Scope affects camera goals only. WBC still controls all 29 motors to balance.
class CameraReferenceScope {
    bool arms_;
public:
    explicit CameraReferenceScope(const std::string& name="wholebody") : arms_(name=="arms") {
        if(name!="wholebody" && name!="whb" && name!="arms")
            throw std::runtime_error("retarget_scope must be wholebody (WHB) or arms");
    }
    const char* name() const { return arms_ ? "Arms" : "WHB"; }
    std::array<Reference,10> apply(std::array<Reference,10> frames) const {
        if(!arms_) return frames;
        for(auto& frame:frames) {
            // A fixed controller stance is independent of camera legs/torso.
            frame.q.head<15>()=standing().head<15>();
            frame.dq.head<15>().setZero();
            for(int i=0;i<15;++i) frame.active[i]=false;
            frame.root=Eigen::Quaterniond::Identity();
            frame.base_tracked=false;
            frame.motion_reference=false;frame.linear_velocity_world.setZero();frame.root_angular_world.setZero();frame.torso.valid=false;
        }
        return frames;
    }
    void restore_stance(std::array<Reference,10>& smoothed, const std::array<Reference,10>& transitioned) const {
        if(!arms_) return;
        // Hidden-joint holding in the camera filter must not pin a neutral
        // startup knee at zero. Preserve the measured-to-standing entry blend.
        for(size_t f=0;f<smoothed.size();++f) {
            smoothed[f].q.head<15>()=transitioned[f].q.head<15>();
            smoothed[f].dq.head<15>()=transitioned[f].dq.head<15>();
        }
    }
};
} // namespace teleop
