// Evaluate the real Unitree joystick DSL against the saved FSM, without DDS.
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include "unitree_joystick_dsl.hpp"
int main(int argc,char** argv) {
    try {
        if(argc<2)throw std::runtime_error("Usage: g1_fsm_keys_fixture STATE [KEY ...]");
        const auto fsm=YAML::LoadFile(std::string(TELEOP_ROOT)+"/deploy/g1/config/config.yaml")["FSM"];
        const std::string source=argv[1];
        if(!fsm["_"][source])throw std::runtime_error("Unknown FSM state");
        unitree::common::UnitreeJoystick joy;
        for(int i=2;i<argc;++i) {
            // Set edges as received on one joystick update. No ROS/DDS objects.
            auto& key=const_cast<unitree::common::KeyBase&>(unitree::common::dsl::GetKey(joy,argv[i]));
            key.pressed=key.on_pressed=true;
        }
        const auto transitions=fsm[source]["transitions"].as<std::map<std::string,std::string>>();
        nlohmann::json matched=nlohmann::json::array();
        for(const auto& [target,condition]:transitions) {
            unitree::common::dsl::Parser parser(condition);
            auto ast=parser.Parse();
            if(unitree::common::dsl::Compile(*ast)(joy))matched.push_back(target);
        }
        std::string target;
        if(std::find(matched.begin(),matched.end(),"Passive")!=matched.end())target="Passive";
        else if(!matched.empty())target=matched[0];
        std::cout<<nlohmann::json({{"target",target},{"matches",matched}}).dump()<<std::endl;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
