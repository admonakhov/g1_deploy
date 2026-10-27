#pragma once
#include "PolicyParams.hpp"

namespace teleop {
// Port of wbc_deploy::clip_pd_torque_positions from policy_deployment_ros2.
class PdTorqueClip {
    bool enabled_=false;
    double fraction_=.98;
    Joints maximum_=Joints::Zero();
public:
    explicit PdTorqueClip(const YAML::Node& cfg){
        if(!cfg)return;
        enabled_=cfg["enabled"].as<bool>(false);fraction_=cfg["fraction"].as<double>(.98);
        if(!std::isfinite(fraction_)||fraction_<=0.||fraction_>1.)throw std::runtime_error("Invalid pd_torque_clipping.fraction");
        if(enabled_)maximum_=joint_vector(cfg["joint_max_torque"],"PD maximum torque");
    }
    Joints apply(Joints target,const RobotState& robot,const PolicyParams& p)const{
        if(!enabled_)return target;
        for(int i=0;i<29;++i){
            if(maximum_[i]<=0.)continue;
            if(std::abs(p.kp[i])<1e-9){target[i]=robot.q[i];continue;}
            const double torque=p.kp[i]*(target[i]-robot.q[i])-p.kd[i]*robot.dq[i],cap=fraction_*maximum_[i];
            if(std::abs(torque)>cap)target[i]=robot.q[i]+(std::clamp(torque,-cap,cap)+p.kd[i]*robot.dq[i])/p.kp[i];
        }
        return target;
    }
};
}
