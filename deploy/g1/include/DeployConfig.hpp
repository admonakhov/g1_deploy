#pragma once
#include <filesystem>
#include <set>
#include <arpa/inet.h>
#include "TrajLocoPolicy.hpp"
#include "WbcPolicy.hpp"
#include "CameraReferenceScope.hpp"
#include "FixedPoseProgram.hpp"
#include "PdTorqueClip.hpp"
#include "GetUpPolicy.hpp"

namespace teleop {
inline void validate_direct_wbc(const YAML::Node& cfg){
    for(const auto key:{"transition_duration_s","reference_smoothing_s","reference_max_speed","max_joint_speed"})
        if(cfg[key]&&cfg[key].as<double>()!=0.)throw std::runtime_error(std::string("Removed WBC reference processing parameter: ")+key);
}
inline YAML::Node load_deployment(const std::filesystem::path& file,const std::filesystem::path& /*project*/,const std::filesystem::path& override_deploy={}){
    const auto canonical=std::filesystem::canonical(file);const auto base=canonical.parent_path();
    auto config=YAML::LoadFile(canonical.string());auto fsm=config["FSM"],enabled=fsm["_"];
    if(!enabled.IsMap()||!enabled["Passive"]||!enabled["FixStand"]||enabled.begin()->first.as<std::string>()!="Passive")throw std::runtime_error("FSM needs Passive first and FixStand");
    const std::map<std::string,std::string> required={{"Passive","Passive"},{"FixStand","SmoothStand"},{"Idle","TrajLoco"},{"Walk","TrajLoco"},{"Run","TrajLoco"},{"GetUp","VqGetUp"},{"Teleop","WbcTeleop"},{"TeleopArms","WbcTeleop"}};
    for(const auto& [name,type]:required)if(!enabled[name]||enabled[name]["type"].as<std::string>(name)!=type)throw std::runtime_error("Missing or wrong required FSM state: "+name);
    auto deploy=YAML::LoadFile((override_deploy.empty()?base/"deploy.yaml":override_deploy).string());
    const auto initial=deploy["initial_state"].as<std::string>("Passive");
    if(!enabled[initial])throw std::runtime_error("Unknown initial_state");
    deploy["initial_state"]=initial;
    deploy["enable_bad_orientation_passive"]=deploy["enable_bad_orientation_passive"].as<bool>(true);
    const double angle=deploy["bad_orientation_limit_rad"].as<double>(1.),age=deploy["max_sensor_age_s"].as<double>(.08);
    if(!std::isfinite(angle)||angle<=0.||angle>M_PI||!std::isfinite(age)||age<=0.||age>5.)throw std::runtime_error("Invalid orientation limit or sensor age");
    deploy["bad_orientation_limit_rad"]=angle;deploy["max_sensor_age_s"]=age;
    std::filesystem::path defaults_path=deploy["robot_defaults_yaml"].as<std::string>("robot_defaults.yaml");
    if(defaults_path.is_relative())defaults_path=base/defaults_path;
    const auto defaults=YAML::LoadFile(defaults_path.string());
    if(!deploy["pd_torque_clipping"])deploy["pd_torque_clipping"]=YAML::Clone(defaults["pd_torque_clipping"]);
    config["deployment"]=YAML::Clone(deploy);
    if(!std::isfinite(deploy["step_dt"].as<double>())||std::abs(deploy["step_dt"].as<double>()-.02)>1e-8||deploy["policy_hz"].as<int>()!=50)throw std::runtime_error("Policies require step_dt=0.02 and policy_hz=50");
    validate_direct_wbc(deploy);
    const double delay=deploy["buffer_delay_s"].as<double>(),timeout=deploy["stream_timeout_s"].as<double>();
    if(!std::isfinite(delay)||delay<0.||!std::isfinite(timeout)||timeout<=delay+.02)throw std::runtime_error("Invalid camera buffer/timeout");
    in_addr address{};const int port=deploy["udp_port"].as<int>();if(port<1||port>65535||inet_pton(AF_INET,deploy["udp_host"].as<std::string>().c_str(),&address)!=1)throw std::runtime_error("Invalid camera UDP endpoint");
    std::set<int> ids;std::map<std::string,std::string> shared_paths;
    for(const auto& e:enabled){const auto name=e.first.as<std::string>(),type=e.second["type"].as<std::string>(name);const int id=e.second["id"].as<int>();
        if(id<=0||!ids.insert(id).second)throw std::runtime_error("Duplicate/invalid FSM ID");
        if(type!="Passive"&&type!="SmoothStand"&&type!="TrajLoco"&&type!="WbcTeleop"&&type!="VqGetUp")throw std::runtime_error("Removed/unsupported policy type: "+type);
        auto state=fsm[name];if(!state.IsMap())throw std::runtime_error("Missing FSM state: "+name);
        if(state["check_posture"]||state["entry_check"])throw std::runtime_error("Use enable_bad_orientation_passive; entry_check/check_posture gates were removed");
        for(const auto& t:state["transitions"])if(!enabled[t.first.as<std::string>()])throw std::runtime_error("FSM transition points to disabled state");
        if(type=="SmoothStand"){
            joint_vector(state["kp"],"FixStand kp",true);joint_vector(state["kd"],"FixStand kd",true);
            FixedPoseProgram validate(state);
            const auto automatic=state["auto_transition"].as<std::string>("");
            if(!automatic.empty()&&(!enabled[automatic]||!state["transitions"][automatic]))throw std::runtime_error("FixStand auto_transition must be in the FSM graph");
        }
        if(type!="TrajLoco"&&type!="WbcTeleop"&&type!="VqGetUp")continue;
        YAML::Node runtime=YAML::Clone(deploy);for(const auto& item:state)runtime[item.first.as<std::string>()]=YAML::Clone(item.second);
        runtime["enable_bad_orientation_passive"]=runtime["enable_bad_orientation_passive"].as<bool>();
        const double state_angle=runtime["bad_orientation_limit_rad"].as<double>();
        if(!std::isfinite(state_angle)||state_angle<=0.||state_angle>M_PI)throw std::runtime_error("Invalid state bad_orientation_limit_rad");
        auto clip=YAML::Clone(deploy["pd_torque_clipping"]);
        if(state["pd_torque_clipping"])for(const auto& field:state["pd_torque_clipping"])clip[field.first.as<std::string>()]=YAML::Clone(field.second);
        runtime["pd_torque_clipping"]=clip;PdTorqueClip validate_clip(clip);
        if(runtime["policy_hz"].as<int>()!=50||std::abs(runtime["step_dt"].as<double>()-.02)>1e-8)throw std::runtime_error("State overrides the trained policy frequency");
        std::filesystem::path directory(state["policy_dir"].as<std::string>());if(directory.is_relative())directory=base/directory;
        directory=std::filesystem::absolute(directory).lexically_normal();runtime["policy_dir"]=directory.string();
        if(shared_paths.count(type)&&shared_paths[type]!=directory.string())throw std::runtime_error("States of the same policy type must share the model bundle");
        const bool first=!shared_paths.count(type);
        shared_paths[type]=directory.string();
        if(type=="VqGetUp"){
            const auto target=state["return_state"].as<std::string>();
            if(!enabled[target]||!state["transitions"][target])throw std::runtime_error("GetUp return_state must be in the FSM graph");
            if(first){GetUpPolicy validate(directory);}
        }else if(type=="TrajLoco"){auto cfg=YAML::LoadFile((directory/"config.yaml").string());const auto params=vq_params(cfg);(void)params;
            TrajLocoPolicy::validate_velocity_limits(state["velocity_limits"]);
            for(const auto relative:{"checkpoints/2026-02-27_18-31-24_decoder.onnx","aux/stand_adjust.onnx","aux/estimator.onnx"})if(!std::filesystem::is_regular_file(directory/relative))throw std::runtime_error("Missing traj_loco model: "+(directory/relative).string());
            runtime["idle_only"]=state["idle_only"].as<bool>(false);
            if(first){TrajLocoPolicy validate(directory);}
        }else{auto cfg=YAML::LoadFile((directory/"params/config.yaml").string());const auto params=wbc_params(cfg);
            validate_direct_wbc(runtime);
            const auto fallback=state["failure_state"].as<std::string>("Walk");
            if(fallback!="Walk"||!enabled[fallback]||!state["transitions"][fallback])throw std::runtime_error("Teleop failure_state must be Walk and present in the FSM graph");
            runtime["failure_state"]=fallback;
            if((params.defaults-standing()).cwiseAbs().maxCoeff()>1e-8)throw std::runtime_error("WBC default/fallback stance mismatch");
            if(!std::filesystem::is_regular_file(directory/"params/policy.onnx"))throw std::runtime_error("Missing WBC model");
            CameraReferenceScope validate(runtime["retarget_scope"].as<std::string>("wholebody"));
            const double height=runtime["wbc_base_height"].as<double>();if(!std::isfinite(height))throw std::runtime_error("Invalid WBC torso reference height");
            if(first){WbcPolicy validate(directory);}
        }
        fsm[name]=runtime;
    }
    if(shared_paths.size()!=3)throw std::runtime_error("traj_loco, WBC and GetUp must be configured");
    return config;
}
} // namespace teleop
