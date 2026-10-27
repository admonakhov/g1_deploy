#pragma once
#include <atomic>
#include "G1Constants.hpp"

namespace teleop {
enum class PolicyFailure { None, Teleop, OtherPolicy, InvalidRobotData };
class PolicyFailureLatch {
    std::atomic<PolicyFailure> value_{PolicyFailure::None};
public:
    void clear(){value_=PolicyFailure::None;}
    void report(PolicyFailure failure){
        if(failure==PolicyFailure::InvalidRobotData){value_=failure;return;}
        // A receiver error must never replace an existing invalid-state fault.
        auto expected=PolicyFailure::None;value_.compare_exchange_strong(expected,failure);
    }
    bool active()const{return value_!=PolicyFailure::None;}
    bool walk()const{return value_==PolicyFailure::Teleop;}
    bool passive()const{return active()&&!walk();}
};

// Recovery starts balanced walking with zero velocity. A held stick must be
// centered for 0.2 seconds before it can request motion again.
class RecoveryInput {
    bool waiting_=false;
    double centered_since_=-1.;
public:
    void reset(bool recovering){waiting_=recovering;centered_since_=-1.;}
    bool waiting()const{return waiting_;}
    Eigen::Vector3d sample(double lx,double ly,double rx,double now){
        Eigen::Vector3d input(lx,ly,rx);
        if(!input.allFinite()||!std::isfinite(now))throw std::runtime_error("Invalid gamepad input");
        if(!waiting_)return input;
        if(input.cwiseAbs().maxCoeff()>.1)centered_since_=-1.;
        else{
            if(centered_since_<0.)centered_since_=now;
            if(now-centered_since_>=.2-1e-9)waiting_=false;
        }
        return Eigen::Vector3d::Zero();
    }
};
}
