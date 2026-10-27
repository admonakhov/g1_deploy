#pragma once
#include <atomic>
#include <mutex>
#include <thread>
#include "FSM/FSMState.h"
#include "TrajLocoPolicy.hpp"
#include "WbcControl.hpp"
#include "UdpInput.hpp"
#include "GetUpPolicy.hpp"
#include "PolicyRecovery.hpp"

class State_Policy:public FSMState{
public:
    enum class Mode { Walking, Teleop, GetUp };
private:
    struct Engine;
    std::shared_ptr<Engine> engine_;
    YAML::Node cfg_;
    Mode mode_;
    bool valid_=false;
public:
    State_Policy(int state,std::string name,Mode mode);
    ~State_Policy();
    void enter()override;void run()override;void exit()override;
    static teleop::RobotState measured(bool check_orientation=true);
};
class State_TrajLoco:public State_Policy{public:State_TrajLoco(int state,std::string name):State_Policy(state,name,Mode::Walking){}};
class State_WbcTeleop:public State_Policy{public:State_WbcTeleop(int state,std::string name):State_Policy(state,name,Mode::Teleop){}};
class State_VqGetUp:public State_Policy{public:State_VqGetUp(int state,std::string name):State_Policy(state,name,Mode::GetUp){}};
REGISTER_FSM(State_TrajLoco)
REGISTER_FSM(State_WbcTeleop)
REGISTER_FSM(State_VqGetUp)
