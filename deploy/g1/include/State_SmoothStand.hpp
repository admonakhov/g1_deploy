#pragma once
#include "State_Policy.hpp"
#include "FixedPoseProgram.hpp"

// Stand is a non-policy state, as in ROS2 deploy_policy. No automatic WBC start.
class State_SmoothStand:public FSMState{
    teleop::FixedPoseProgram program_;
    teleop::Joints kp_,kd_;
    bool fault_=false,completed_=false;
public:
    State_SmoothStand(int state,std::string name):FSMState(state,name),program_(param::config["FSM"][name]),
        kp_(teleop::joint_vector(param::config["FSM"][name]["kp"],"FixStand kp",true)),
        kd_(teleop::joint_vector(param::config["FSM"][name]["kd"],"FixStand kd",true)){
        registered_checks.insert(registered_checks.begin(),{[this]{return fault_;},FSMStringMap.right.at("Passive")});
        const auto automatic=param::config["FSM"][name]["auto_transition"].as<std::string>("");
        if(!automatic.empty())registered_checks.push_back({[this]{
            if(completed_||!program_.finished(teleop::steady_seconds()))return false;
            completed_=true;return true;
        },FSMStringMap.right.at(automatic)});
    }
    void enter()override{
        fault_=completed_=false;
        try{
            const auto robot=State_Policy::measured(false);program_.begin(robot,teleop::steady_seconds());
            for(int i=0;i<29;++i){auto& m=lowcmd->msg_.motor_cmd()[i];m.q()=robot.q[i];m.kp()=kp_[i];m.kd()=kd_[i];m.dq()=m.tau()=0.;}
            spdlog::info("FixStand: policy disabled; linear joint PD transition ts/qs from measured pose, any orientation");
        }catch(const std::exception& e){fault_=true;spdlog::error("Cannot start FixStand: {}",e.what());}
    }
    void run()override{
        if(fault_)return;
        try{
            State_Policy::measured(false);
            const auto q=program_.sample(teleop::steady_seconds());
            for(int i=0;i<29;++i)lowcmd->msg_.motor_cmd()[i].q()=q[i];
        }catch(const std::exception& e){fault_=true;spdlog::error("FixStand data: {}; switching to Passive",e.what());}
    }
};
REGISTER_FSM(State_SmoothStand)
