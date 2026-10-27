#pragma once
#include "PolicyParams.hpp"
#include "RobotChecks.hpp"

namespace teleop {
// ROS2 deploy_policy::enter_fixstand + linear_interpolate, in SDK joint order.
class FixedPoseProgram {
    std::vector<double> times_;
    std::vector<Joints> configured_,poses_;
    std::vector<bool> empty_;
    double began_=0.;
public:
    explicit FixedPoseProgram(const YAML::Node& cfg){
        times_=cfg["ts"].as<std::vector<double>>();const auto qs=cfg["qs"];
        if(times_.size()<2||!qs.IsSequence()||qs.size()!=times_.size())throw std::runtime_error("FixStand requires matching ts/qs waypoints");
        for(size_t i=0;i<times_.size();++i){
            if(!std::isfinite(times_[i])||times_[i]<0.||(i&&times_[i]<=times_[i-1]))throw std::runtime_error("FixStand ts must increase");
            const bool empty=qs[i].IsSequence()&&qs[i].size()==0;empty_.push_back(empty);
            configured_.push_back(empty?Joints::Zero().eval():joint_vector(qs[i],"FixStand waypoint"));
        }
    }
    void begin(const RobotState& robot,double now){
        validate_robot_state(robot,false);if(!std::isfinite(now))throw std::runtime_error("Invalid FixStand time");
        poses_=configured_;poses_[0]=robot.q;
        for(size_t i=1;i<poses_.size();++i)if(empty_[i])poses_[i]=poses_[i-1];
        began_=now;
    }
    bool finished(double now)const{return now-began_>=times_.back();}
    Joints sample(double now)const{
        if(poses_.empty()||!std::isfinite(now))throw std::runtime_error("FixStand not initialized or invalid time");
        const double t=now-began_;
        if(t<=times_.front())return poses_.front();
        if(t>=times_.back())return poses_.back();
        for(size_t i=0;i+1<times_.size();++i)if(t<=times_[i+1]){
            const double a=(t-times_[i])/(times_[i+1]-times_[i]);return (1.-a)*poses_[i]+a*poses_[i+1];
        }
        return poses_.back();
    }
};
}
